from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
import torch
from .datasets import ClipDataset, ClipDatasetRanked
from .transforms import build_transformer
from .sampler import build_sampler
from .metrics import build_evaluator
from .imagenet_dataloader import build_common_augmentation
from easydict import EasyDict


def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    image_ids = [_['image_id'] for _ in batch]
    filenames = [_['filename'] for _ in batch]
    if type(batch[0]['image']) == list:
        images = [torch.stack([_['image'][0] for _ in batch]), torch.stack([_['image'][1] for _ in batch]), \
                  torch.stack([_['image'][2] for _ in batch]), torch.stack([_['image'][3] for _ in batch])]
              
    else:
        images = torch.stack([_['image'] for _ in batch])
    # print(batch[0].keys())
    # print(batch[0]['label'])
    # print(batch[0]['label_name'])
    # print(batch[0]['tag'])

    labels = torch.as_tensor([_.get('label', -1)
                              for _ in batch], dtype=torch.long)
    label_names = [_.get('label_name', None) for _ in batch]
    captions = [_.get('caption', []) for _ in batch]
    tags = [_.get('tag', []) for _ in batch]

    # sources = [_['source'] for _ in batch]
    # flipped = [_['flipped'] for _ in batch]
    # neg_targets = [_.get('neg_target', 0) for _ in batch]
    # image_sources = [_.get('image_source', 0) for _ in batch]
    # meta_info = [_.get('meta_info', {}) for _ in batch]
    # gt_bboxes = [_.get('gt_bboxes', None) for _ in batch]
    # gt_scores = [_.get('gt_scores', None) for _ in batch]
    # gt_ignores = [_.get('gt_ignores', None) for _ in batch]
    # gt_keyps = [_.get('gt_keyps', None) for _ in batch]
    # gt_masks = [_.get('gt_masks', None) for _ in batch]
    # gt_grids = [_.get('gt_grids', None) for _ in batch]
    # gt_semantic_seg = [_.get('gt_semantic_seg', None) for _ in batch]
    # padded_images = self.pad(images)

    output = EasyDict({
        'image_ids': image_ids,
        'filenames': filenames,
        'images': images,
        'captions': captions,
        'tags': tags,
    })

    output['labels'] = labels if labels[0] is not None else None
    output['label_names'] = label_names if label_names[0] is not None else None
    # output['captions'] = captions if captions[0] is not None else None
    # output['gt_masks'] = gt_masks if gt_masks[0] is not None else None
    # output['gt_grids'] = gt_grids if gt_grids[0] is not None else None
    # output['gt_scores'] = gt_scores if gt_scores[0] is not None else None
    # if gt_semantic_seg[0] is not None:
    #     output['gt_semantic_seg'] = self.pad(gt_semantic_seg)
    return output


def build_clip_dataloader(data_type, cfg_dataset):
    """
    arguments:
        - data_type: 'train', 'test', 'val'
        - cfg_dataset: configurations of dataset
    """
    assert data_type in cfg_dataset
    # build transformer
    image_reader = cfg_dataset[data_type].get('image_reader', {})
    if isinstance(cfg_dataset[data_type]['transforms'], list):
        transformer = build_transformer(cfgs=cfg_dataset[data_type]['transforms'],
                                        image_reader=image_reader)
    else:
        transformer = build_common_augmentation(
            cfg_dataset[data_type]['transforms']['type'])
    # build evaluator
    evaluator = None
    if data_type == 'test' and cfg_dataset[data_type].get('evaluator', None):
        evaluator = build_evaluator(cfg_dataset[data_type]['evaluator'])
    # build dataset
    if cfg_dataset['type'] == 'clip':
        CurrDataset = ClipDataset
        if cfg_dataset[data_type].get('use_ranked', False):
            CurrDataset = ClipDatasetRanked
            cfg_dataset[data_type]['sampler']['type'] = 'ranked_iteration'
    else:
        raise NotImplementedError

    if cfg_dataset['read_from'] == 'osg':
        dataset = CurrDataset(
            root_dir='',
            meta_file=cfg_dataset[data_type]['meta_file'],
            transform=transformer,
            read_from='osg',
            evaluator=evaluator,
            image_reader_type=image_reader.get('type', 'pil'),
            osg_server=cfg_dataset[data_type]['osg_server'],
            label_texts_ensemble=cfg_dataset[data_type].get('label_texts_ensemble', 'none')
        )
    else:
        more_args = {}
        if cfg_dataset[data_type].get('offset_file_prefix', ''):
            more_args['offset_file_prefix']=cfg_dataset[data_type].offset_file_prefix
        dataset = CurrDataset(
            root_dir=cfg_dataset[data_type]['root_dir'],
            meta_file=cfg_dataset[data_type]['meta_file'],
            transform=transformer,
            read_from=cfg_dataset['read_from'],
            evaluator=evaluator,
            image_reader_type=image_reader.get('type', 'pil'),
            server_cfg=cfg_dataset[data_type].get('server_cfg', {}),
            fseek=cfg_dataset[data_type].get('fseek',False),
            label_texts_ensemble=cfg_dataset[data_type].get('label_texts_ensemble', 'none'),
            **more_args
        )
    # initialize kwargs of sampler
    cfg_dataset[data_type]['sampler'].setdefault('kwargs', {}) #'last_iter': cfg_dataset['last_iter']})
    cfg_dataset['dataset'] = dataset
    # build sampler
    sampler = build_sampler(cfg_dataset[data_type]['sampler'], cfg_dataset)
    if data_type == 'train' and cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}
    # build dataloader
    loader = DataLoader(dataset=dataset,
                        batch_size=cfg_dataset['batch_size'],
                        shuffle=False if sampler is not None else True,
                        num_workers=cfg_dataset['num_workers'],
                        pin_memory=cfg_dataset['pin_memory'],
                        sampler=sampler,
                        collate_fn=_collate_fn)
    return {'type': data_type, 'loader': loader}
