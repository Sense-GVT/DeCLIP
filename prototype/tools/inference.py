import os
import argparse
import pprint
import torch
import json
import cv2
import numpy as np
import linklink as link

import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from easydict import EasyDict
from torch.autograd import Variable

from prototype.solver.cls_solver import ClsSolver
from prototype.utils.dist import link_dist
from prototype.utils.misc import makedir, create_logger, get_logger, modify_state
from prototype.data import build_imagenet_test_dataloader
from prototype.data import build_custom_dataloader


class Inference(ClsSolver):
    def __init__(self, config):
        self.image_dir = config["image_dir"]
        self.meta_file = config.get("meta_file", "")

        self.output = config.get("output", "inference_results")
        self.recover = config.get("recover", "")
        self.cam = config.get("cam", False)
        self.visualize = config.get("visualize", False)
        self.sample = config.get("sample", -1)
        self.feature_name = config.get("name", "module.layer4")
        if "module" not in self.feature_name:
            self.feature_name = "module." + self.feature_name
        self.feature = None
        self.gradient = None
        super(Inference, self).__init__(config["config"])

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.result_path = os.path.abspath(self.output)
        makedir(self.path.result_path)
        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if self.recover != "":
            self.state = torch.load(self.recover, 'cpu')
            self.logger.info(f"Recovering from {self.recover}, keys={list(self.state.keys())}")

        elif hasattr(self.config.saver, 'pretrain'):
            self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
            self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
        else:
            self.state = {}
            self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def build_data(self):
        self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        self.config.data.last_iter = self.state['last_iter']

        root_dir, input_file = self.generate_custom_data()
        self.config.data.test.root_dir = root_dir
        self.config.data.test.meta_file = input_file

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            self.val_data = build_custom_dataloader('test', self.config.data)

    def generate_custom_data(self):

        if self.meta_file != "" and os.path.exists(self.meta_file):
            return self.image_dir, self.meta_file

        input_file = os.path.join(self.output, "tmp_meta.json")
        image_dir = self.image_dir

        if os.path.isfile(self.image_dir):
            image_dir = os.path.abspath(os.path.dirname(self.image_dir))
            if self.dist.rank == 0:
                with open(input_file, "w") as output:
                    output.write(json.dumps({"filename": os.path.basename(self.image_dir)},
                                            ensure_ascii=False) + '\n')
        else:
            if self.dist.rank == 0:
                with open(input_file, "w") as output:
                    meta_list = []
                    for root, dirs, files in os.walk(self.image_dir, topdown=False):
                        for name in files:
                            abs_path = os.path.join(root, name)
                            meta_list.append(abs_path[len(self.image_dir):].lstrip("/"))
                    sample_num = len(meta_list)
                    if 0 < self.sample < 1:
                        sample_num = int(self.sample * sample_num)
                    elif self.sample > 1:
                        sample_num = max(sample_num, self.sample)

                    for idx in range(sample_num):
                        output.write(json.dumps({"filename": meta_list[idx], "label_name": "abs", "label": 1},
                                                ensure_ascii=False) + '\n')

        link.barrier()
        return image_dir, input_file

    def paint(self, filename, pred, label, outdir):

        num = len(filename)

        for idx in range(num):
            ax, fig, h, w = self.get_axis(filename[idx])
            self.paint_one_image(pred[idx], label[idx], ax, h, w)
            out_name = os.path.join(outdir, os.path.basename(filename[idx]))
            fig.savefig(out_name, dpi=200)
            plt.close('all')

    @staticmethod
    def paint_one_image(pred, label, ax, h, w):
        font_sz = max(min(np.log(h) / np.log(100), np.log(w) / np.log(100)), 1)
        x1 = w // 8
        y1 = h // 8

        ax.text(
            x1, y1,
            f"cls {int(label)}, score:{pred[int(label)]:.3f}",
            fontsize=font_sz + 10,
            family='serif',
            color="r"
        )

    @staticmethod
    def get_axis(img_path):
        assert os.path.exists(img_path), f"check img file path, {img_path}"
        img = cv2.imread(img_path)[:, :, (2, 1, 0)]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(img.shape[1] / 200, img.shape[0] / 200)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(img)
        return ax, fig, img.shape[0], img.shape[1]

    def inference(self):
        self.model.eval()

        res_file = os.path.join(self.output, f'results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.val_data['loader']):
            input = batch['image']
            input = input.cuda().half() if self.fp16 else input.cuda()

            # compute output
            logits = self.model(input)

            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds.detach()})
            batch.update({'score': scores.detach()})
            # save prediction information
            if self.cam:
                heatmap = self.gradCam(input)
                for idx in range(len(heatmap)):
                    basename = os.path.basename(batch["filename"][idx])
                    ext = basename.split(".")[-1]
                    basename = basename.replace("." + ext, "_cam" + "." + ext)

                    heatmap[idx].save(os.path.join(self.output, basename))

            if self.visualize:
                self.paint(batch["filename"], scores, preds, self.output)
            self.val_data['loader'].dataset.dump(writer, batch)

        writer.close()
        link.barrier()

        return

    def save_feature(self, module, input, output):
        self.feature = output

    def save_gradient(self, module, grad_in, grad_out):

        self.gradient = grad_out[0].detach()

    def gradCam(self, x):
        model = self.model.eval()
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x, requires_grad=True)
        heat_maps = []
        for i in range(datas.size(0)):
            feature = datas[i].unsqueeze(0)

            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            for name, module in self.model.named_modules():

                if name == self.feature_name:
                    module.register_forward_hook(self.save_feature)
                    module.register_backward_hook(self.save_gradient)

            feature = model(feature)

            classes = F.softmax(feature, dim=1)

            one_hot, _ = classes.max(dim=-1)
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)

            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)

            mask = cv2.resize(mask.data.cpu().numpy().astype(np.float32), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToPILImage()(transforms.ToTensor()(
                cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB))))
        return heat_maps


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Inference Solver')
    parser.add_argument('--config', required=True, type=str, help="Prototype task yaml")
    parser.add_argument('--recover', default="", help="Recover model path to visuazlie")

    parser.add_argument('-i', '--image_dir', required=True, dest="image_dir", type=str,
                        help="The image dir that you want to visuazlie.")
    parser.add_argument('-m', '--meta_file', required=False, dest="meta_file", type=str,
                        help="The prototype custom meta file that you want to visuazlie. "
                             "If this argument are not provide, we will visualize the images in {image_dir}")
    parser.add_argument('-o', '--output', default="./inference_resuts", dest="output",
                        help="the folder where results file or images will be saved.")
    parser.add_argument('--visualize', default=True, help="Whether paint class and score on images to visualize.")
    parser.add_argument('--sample', default=-1, type=float,
                        help="if gived number -1, remain all results. if 0 < gived number <=1, "
                             "sample {gived number * len(images_dir)} images, "
                             "if gived number > 1, sample gived number images.")
    parser.add_argument('--cam', default=False,
                        help="Whether save gradcam results. See https://arxiv.org/abs/1610.02391 for details.")
    parser.add_argument('--name', default="module.layer4",
                        help="the last feature extractor layer name you want to visualize gradcam results, "
                             "e.g. 'layer4' in resnet series.")

    args = parser.parse_args()
    # build solver
    inference_helper = Inference(args.__dict__)
    inference_helper.inference()


if __name__ == '__main__':
    main()
