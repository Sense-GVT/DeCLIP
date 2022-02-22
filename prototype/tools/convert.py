import os
import argparse
from easydict import EasyDict
import shutil
import numpy as np
import yaml
import json

import torch
import torch.nn as nn
import linklink as link

from prototype.utils.dist import link_dist
from prototype.solver.cls_solver import ClsSolver
from prototype.utils.misc import parse_config, load_state_model
from prototype.utils.nnie_helper import generate_nnie_config


class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x


class KestrelSolver(ClsSolver):

    def __init__(self, config_file, recover=''):
        self.config_file = config_file
        self.recover = recover
        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()
        # 'recover' only for convert
        if self.recover:
            self.logger.info(f"Recover exist! Again Recovering from {self.recover}")
            recover_state = torch.load(self.recover, 'cpu')
            load_state_model(self.model, recover_state['model'])
        if self.config.to_kestrel.get('add_softmax'):
            self.model = Wrapper(self.model)

    def to_caffe(self, save_prefix='model', input_size=None):
        try:
            from spring.nart.tools import pytorch
        except ImportError:
            print('Install Spring NART first!')

        with pytorch.convert_mode():
            pytorch.convert(self.model.float(),
                            [(3, self.config.data.input_size,
                              self.config.data.input_size)],
                            filename=save_prefix,
                            input_names=['data'],
                            output_names=['out'])

    def to_nnie(self, nnie_cfg, config, prototxt, caffemodel, model_name):
        nnie_cfg_path = generate_nnie_config(nnie_cfg, config)
        nnie_cmd = 'python -m spring.nart.switch -c {} -t nnie {} {}'.format(
            nnie_cfg_path, prototxt, caffemodel)

        os.system(nnie_cmd)
        assert os.path.exists("parameters.json")
        with open("parameters.json", "r") as f:
            params = json.load(f)
        params["model_files"]["net"]["net"] = "engine.bin"
        params["model_files"]["net"]["backend"] = "kestrel_nart"
        with open("parameters.json", "w") as f:
            json.dump(params, f, indent=2)
        tar_cmd = 'tar cvf {} engine.bin engine.bin.json meta.json meta.conf parameters.json category_param.json'.\
            format(model_name + "_nnie.tar")
        os.system(tar_cmd)
        self.logger.info(f"generate {model_name + '_nnie.tar'} done!")

    def refactor_config(self):
        '''Prepare configuration for kestrel classifier model. For details:
        https://confluence.sensetime.com/display/VIBT/nart.tools.kestrel.classifier
        '''
        kestrel_config = EasyDict()
        kestrel_config['pixel_means'] = self.config.to_kestrel.get('pixel_means', [123.675, 116.28, 103.53])
        kestrel_config['pixel_stds'] = self.config.to_kestrel.get('pixel_stds', [58.395, 57.12, 57.375])

        kestrel_config['is_rgb'] = self.config.to_kestrel.get('is_rgb', True)
        kestrel_config['save_all_label'] = self.config.to_kestrel.get('save_all_label', True)
        kestrel_config['type'] = self.config.to_kestrel.get('type', 'ImageNet')

        if self.config.get('to_kestrel') and self.config.to_kestrel.get('class_label'):
            kestrel_config['class_label'] = self.config.to_kestrel['class_label']
        else:
            kestrel_config['class_label'] = {}
            kestrel_config['class_label']['imagenet'] = {}
            kestrel_config['class_label']['imagenet']['calculator'] = 'bypass'
            num_classes = self.config.model.get('num_classes', 1000)
            kestrel_config['class_label']['imagenet']['labels'] = [
                str(i) for i in np.arange(num_classes)]
            kestrel_config['class_label']['imagenet']['feature_start'] = 0
            kestrel_config['class_label']['imagenet']['feature_end'] = num_classes - 1

        self.kestrel_config = kestrel_config

    def to_kestrel(self, save_to=None):
        prefix = 'model'
        self.logger.info('Converting Model to Caffe...')
        if self.dist.rank == 0:
            self.to_caffe(prefix)

        link.synchronize()
        self.logger.info('To Caffe Done!')

        prototxt = '{}.prototxt'.format(prefix)
        caffemodel = '{}.caffemodel'.format(prefix)
        version = self.config.to_kestrel.get('version') if self.config.to_kestrel.get('version') else '1.0.0'
        model_name = self.config.to_kestrel.get('model_name') if self.config.to_kestrel.get('model_name') \
            else self.config.model.type

        kestrel_model = '{}_{}.tar'.format(model_name, version)
        to_kestrel_yml = 'temp_to_kestrel.yml'
        self.refactor_config()

        with open(to_kestrel_yml, 'w') as f:
            yaml.dump(json.loads(json.dumps(self.kestrel_config)), f)

        cmd = 'python -m spring.nart.tools.kestrel.classifier {} {} -v {} -c {} -n {}'.format(
            prototxt, caffemodel, version, to_kestrel_yml, model_name)

        self.logger.info('Converting Model to Kestrel...')
        if self.dist.rank == 0:
            os.system(cmd)

        link.synchronize()
        self.logger.info('To Kestrel Done!')

        if save_to is None:
            save_to = kestrel_model
        else:
            save_to = os.path.join(save_to, kestrel_model)

        shutil.move(kestrel_model, save_to)
        link.synchronize()
        self.logger.info('Save kestrel model to: {}'.format(save_to))

        # convert model to nnie
        nnie_cfg = self.config.to_kestrel.get('nnie', None)
        if nnie_cfg is not None:
            self.logger.info('Converting Model to NNIE...')
            if self.dist.rank == 0:
                self.to_nnie(nnie_cfg, self.config, prototxt, caffemodel, model_name)
            link.synchronize()
            self.logger.info('To NNIE Done!')


@link_dist
def main():
    parser = argparse.ArgumentParser(description='caffe/kestrel solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--recover', type=str, default='')

    args = parser.parse_args()

    # build or recover solver
    solver = KestrelSolver(args.config, recover=args.recover)
    # to caffe and to kestrel
    solver.to_kestrel()


if __name__ == '__main__':
    main()
