import os.path as osp
import json
from .base_dataset import BaseDataset
from prototype.data.image_reader import build_image_reader


class ImageNetDataset(BaseDataset):
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader (:obj:`str`): reader type 'pil' or 'ks'

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """
    def __init__(self, root_dir, meta_file, transform=None,
                 read_from='mc', evaluator=None, image_reader_type='pil'):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.initialized = False

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            filename, label = line.rstrip().split()
            self.metas.append({'filename': filename, 'label': label})

        super(ImageNetDataset, self).__init__(root_dir=root_dir,
                                              meta_file=meta_file,
                                              read_from=read_from,
                                              transform=transform,
                                              evaluator=evaluator)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = osp.join(self.root_dir, curr_meta['filename'])
        label = int(curr_meta['label'])
        # add root_dir to filename
        curr_meta['filename'] = filename
        img_bytes = self.read_file(curr_meta)
        img = self.image_reader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {
            'image': img,
            'label': label,
            'image_id': idx,
            'filename': filename
        }
        return item

    def dump(self, writer, output):
        prediction = self.tensor2numpy(output['prediction'])
        label = self.tensor2numpy(output['label'])
        score = self.tensor2numpy(output['score'])

        if 'filename' in output:
            # pytorch type: {'image', 'label', 'filename', 'image_id'}
            filename = output['filename']
            image_id = output['image_id']
            for _idx in range(prediction.shape[0]):
                res = {
                    'filename': filename[_idx],
                    'image_id': int(image_id[_idx]),
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        else:
            # dali type: {'image', 'label'}
            for _idx in range(prediction.shape[0]):
                res = {
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()
