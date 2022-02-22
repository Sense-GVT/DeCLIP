# -*- coding: utf-8 -*-
import torch
import json
import os.path as osp
import requests
import numpy as np
import time
from typing import List
from .base_dataset import BaseDataset
from prototype.data.image_reader import build_image_reader
import linklink as link
import random
import os

def joinpath(root, suffix):
    if root != '' and suffix[0] == '/':
        suffix = suffix[1:]
    if root[-1] == ':':
        return root + suffix
    return osp.join(root,suffix)


class ClipDataset(BaseDataset):
    """
    Clip Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - osg_server (:obj:`str`): '10.198.3.28:30080/components/osg-default/v1'

    Metafile example::
        "{"filename": "n01440764/n01440764_10026.JPEG", "label": 0, "label_name": "dog"}\n"
    """

    def __init__(self, root_dir, meta_file, transform=None,
                 read_from='mc', evaluator=None, image_reader_type='pil',
                 server_cfg={}, fseek=False, label_texts_ensemble='none',rank_dataset=False):

        if not isinstance(meta_file, List):
            meta_file = [meta_file]
        if not isinstance(root_dir, List):
            root_dir = [root_dir]

        self.meta_file = meta_file
        self.root_dir = root_dir
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.server_cfg = server_cfg
        self.fseek = fseek
        self.rank_dataset = rank_dataset
        self.initialized = False
        self.label_texts_ensemble = label_texts_ensemble
        self.num = 0

        self.metas = []
        if self.rank_dataset: 
            self.read_meta_files(root_dir, meta_file)

        elif self.fseek:
            self.line_offsets = []
            for each_meta_file in meta_file:
                line_offset = []
                offset = 0
                with open(each_meta_file) as f:
                    for line in f:
                        line_offset.append(offset)
                        offset += len(line.encode('UTF-8'))
                    f.close()
                self.num += len(line_offset)
                self.line_offsets.append(line_offset)
        elif len(server_cfg) == 0:
            # read from local file
            for rd, each_meta_file in zip(root_dir, meta_file):
                with open(each_meta_file) as f:
                    lines = f.readlines()

                self.num += len(lines)

                for line in lines:
                    info = json.loads(line)
                    filename = joinpath(rd, info['filename'])
                    # add root_dir to filename
                    info['filename'] = filename
                    self.metas.append(info)
        else:
            # read from http server
            self.server_ip = server_cfg['ip']
            self.server_port = server_cfg['port']
            if isinstance(self.server_ip, str):
                self.server_ip = [self.server_ip]
            if isinstance(self.server_port, int):
                self.server_port = [self.server_port]
            assert len(self.server_ip) == len(self.server_port), \
                'length of ips should equal to the length of ports'

            self.num = int(requests.get('http://{}:{}/get_len'.format(
                self.server_ip[0], self.server_port[0])).json())

        super(ClipDataset, self).__init__(root_dir=root_dir,
                                          meta_file=meta_file,
                                          read_from=read_from,
                                          transform=transform,
                                          evaluator=evaluator)

    def __len__(self):
        return self.num

    def _str2list(self, x):
        if type(x) is list:
            return x
        elif type(x) is str:
            return [x]
        else:
            raise RuntimeError(
                "unknown value for _str2list: {}".format(type(x)))

    def _load_meta(self, idx):
        if self.rank_dataset:
            if self.fseek:
                source_id = 0
                while idx >= len(self.line_offsets[source_id]):
                    idx -= len(self.line_offsets[source_id])
                    source_id += 1 #fixed
                with open(self.meta_file[source_id]) as f:
                    f.seek(self.line_offsets[source_id][idx])
                    line = f.readline()
                    meta = json.loads(line)
                    filename = joinpath(self.root_dir[source_id], meta['filename'])
                    meta['filename'] = filename
                    f.close()
                return meta
            else:
                return self.metas[idx]
        elif self.fseek:
            source_id = 0
            while idx >= len(self.line_offsets[source_id]):
                idx -= len(self.line_offsets[source_id])
                source_id += 1 #fixed
            with open(self.meta_file[source_id]) as f:
                f.seek(self.line_offsets[source_id][idx])
                line = f.readline()
                meta = json.loads(line)
                filename = joinpath(self.root_dir[source_id], meta['filename'])

                meta['filename'] = filename
                f.close()
            return meta
        elif len(self.server_cfg) == 0:
            return self.metas[idx]
        else:
            while True:
                # random select a server ip
                rdx = np.random.randint(len(self.server_ip))
                r_ip, r_port = self.server_ip[rdx], self.server_port[rdx]
                # require meta information
                try:
                    meta = requests.get(
                        'http://{}:{}/get/{}'.format(r_ip, r_port, idx), timeout=500).json()
                    # 'http://{}:{}/get/{}'.format(r_ip, r_port, idx)).json()
                    break
                except Exception:
                    # continue
                    # time.sleep(0.005)
                    time.sleep(0.00005)
            # print("******************source_id", meta['source_id'])
            # print("******************len", len(self.root_dir))
            # print(meta)
            # print(self.root_dir)
            meta['filename'] = joinpath(
                self.root_dir[meta['source_id']], meta['filename'])
            return meta

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = curr_meta['filename']
        # 'label' and 'label_name' is optional for CLIP
        label = int(curr_meta['label']) if 'label' in curr_meta else -1
        label_name = curr_meta['label_name'] if 'label_name' in curr_meta else None
        caption = self._str2list(
            curr_meta['caption']) if 'caption' in curr_meta else []
        tag = self._str2list(curr_meta['tag']) if 'tag' in curr_meta else []

        try:
            if ''.join(caption) == '':
                raise RuntimeError('str is nothing')
            # if self.is_contains_chinese(caption):
            #     print('***************************************', caption)
            assert self.is_contains_chinese(caption) == False
            img_bytes = self.read_file(curr_meta)
            img = self.image_reader(img_bytes, filename)
            if self.transform is not None:
                img = self.transform(img)

            item = {
                'image_id': idx,
                'filename': filename,
                'image': img,
                'label': label,
                'label_name': label_name,
                'caption': caption,
                'tag': tag
            }
            return item
        except Exception as e:
            # idx = 0
            # return self.__getitem__(idx)
            print(repr(e), filename, caption, flush=True)
            # raise e
            return self.__getitem__(np.random.randint(self.num))

    def is_contains_chinese(self, strs):
        for _str in strs:
            for _char in _str:
                if '\u4e00' <= _char <= '\u9fa5':
                    return True
        return False

    #def _get_label_text(self, text):
    #    label_text = ['a photo of ' + text + '.']
    #    if self.label_texts_ensemble == 'none':
    #        return label_text
    #    elif self.label_texts_ensemble == 'simple':
    #        label_text.extend(['a photo of small' + text + '.',
    #                           'a photo of big ' + text + '.',
    #                           'a picture of small' + text + '.',
    #                           'a picture of big' + text + '.',
    #                           ])
    #        return label_text
    #    else:
    #        return label_text
    def _get_label_text(self, text):
        # label_text = ['a photo of ' + text + '.']
        if self.label_texts_ensemble == 'prompt6':
            f = f'{osp.abspath(os.getcwd())}/../../../../prototype/data/datasets/prompts/query_pattern_prompt6'
        elif self.label_texts_ensemble == 'prompt8':
            f = f'{osp.abspath(os.getcwd())}/../../../../prototype/data/datasets/prompts/query_pattern_prompt8'
        elif self.label_texts_ensemble == 'prompt80':
            f = f'{osp.abspath(os.getcwd())}/../../../../prototype/data/datasets/prompts/query_pattern_prompt80'
        elif self.label_texts_ensemble == 'cc':
            return [text]
        elif 'file:' in self.label_texts_ensemble:
            f = self.label_texts_ensemble[5:]
        elif self.label_texts_ensemble == 'simple':
            f = f'{osp.abspath(os.getcwd())}/../../../../prototype/data/datasets/prompts/query_pattern_prompt1'
        else:
            raise NotImplementedError(self.label_texts_ensemble)
        label_text = []
        with open(f) as fin:
            for line in fin.readlines():
                label_text.append(line.strip().replace('{0}', text))
        return label_text

    def get_label_texts(self,):
        label_to_name = {}
        for curr_meta in self.metas:
            label = int(curr_meta['label']) if 'label' in curr_meta else None
            label_name = curr_meta['label_name'] if 'label_name' in curr_meta else None
            if label is not None and label_name is not None:
                label_to_name[label] = label_name
        labels = list(label_to_name.keys())
        labels.sort()

        label_texts = []
        label_text_len = []
        for label in labels:
            label_name = label_to_name[label]
            label_text = self._get_label_text(label_name)
            label_texts.extend(label_text)
            label_text_len.append(len(label_text))

        all_len = sum(label_text_len)
        offset = 0
        label_num = len(labels)
        label_texts_ensemble_matrix = torch.eye(label_num)

        # label_texts_ensemble_matrix = torch.zeros(all_len, label_num)
        # for lbl, ltl in enumerate(label_text_len):
        #     label_texts_ensemble_matrix[offset: offset + ltl, lbl] = 1
        #     offset += ltl

        return label_texts, label_texts_ensemble_matrix

    def dump(self, writer, output):
        filenames = output['filenames']
        image_ids = output['image_ids']
        label_names = output['label_names']
        captions = output['captions']
        tags = output['tags']
        prediction = self.tensor2numpy(output['prediction'])
        score = self.tensor2numpy(output['score'])
        labels = self.tensor2numpy(output['labels'])
        for _idx in range(len(filenames)):
            res = {
                'image_id': int(image_ids[_idx]),
                'filename': filenames[_idx],
                'label': int(labels[_idx]),
                'label_name': label_names[_idx],
                'caption': captions[_idx],
                'tag': tags[_idx],
                'prediction': int(prediction[_idx]),
                'score': [float('%.8f' % s) for s in score[_idx]]
            }
            writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()


class ClipDatasetRanked(ClipDataset):
    def __init__(self, root_dir, meta_file, transform=None,
                 read_from='mc', evaluator=None, image_reader_type='pil',
                 server_cfg={}, fseek=False, label_texts_ensemble='none',rank_dataset=True,
                 offset_file_prefix=''):
        self.world_size = link.get_world_size()
        self.rank = link.get_rank()
        self.fseek = fseek
        self.offset_file_prefix = offset_file_prefix
        super(ClipDatasetRanked, self).__init__(root_dir, meta_file, transform,
                                                read_from, evaluator, image_reader_type, server_cfg, fseek, label_texts_ensemble, rank_dataset)

    def read_meta_files(self, root_dir, meta_files):
        if self.fseek:
            ranked_num = [0 for _ in range(self.world_size)]
            self.ranked_line_offsets = [[] for _ in range(self.world_size)]
            import time
            star_time = time.time()
            seed_num, number = 0, 0
            random.seed(seed_num)
            for each_meta_file in meta_files:
                link.barrier()
                if self.rank == 0:
                    print('[Rank %d] Solve offset of file'%self.rank, each_meta_file, flush=True)
                offset_file = None
                ranked_line_offset = [[] for _ in range(self.world_size)]

                if self.offset_file_prefix:
                    offset_file = os.path.join(self.offset_file_prefix, each_meta_file.replace('/', '_').replace('\\','_'))
                    offset_ranked_dic = os.path.join(self.offset_file_prefix, 'ranked_offsets')
                    offset_ranked_file_dic = os.path.join(offset_ranked_dic, each_meta_file.replace('/', '_').replace('\\','_').replace('.', '_'))
                    offset_ranked_file_world_dic = os.path.join(offset_ranked_file_dic, f'{self.world_size}')
                    offset_ranked_file = os.path.join(offset_ranked_file_world_dic, f'{self.rank}.npy')
                    if self.rank == 0:
                        os.makedirs(offset_ranked_dic, exist_ok=True)
                        os.makedirs(offset_ranked_file_dic, exist_ok=True)
                        os.makedirs(offset_ranked_file_world_dic, exist_ok=True)
                link.barrier()
                if offset_ranked_file and os.path.exists(offset_ranked_file):
                    ranked_line_offset[self.rank] = np.load(offset_ranked_file).tolist()
                    ranked_num[self.rank] += len(ranked_line_offset[self.rank])
                else:
                    # import ipdb
                    # ipdb.set_trace()
                    if offset_file and os.path.exists(offset_file):
                        if self.rank == 0:
                            print('[Rank %d] Using Offset File'%self.rank, offset_file, flush=True)
                        with open(offset_file, 'rb') as f:
                            while True:
                                ch = f.read(6)
                                if not ch: break
                                offset = int.from_bytes(ch, byteorder='little')
                                number += 1
                                random_rank = random.randint(0, self.world_size-1)
                                if self.rank == random_rank:
                                    ranked_line_offset[self.rank].append(offset)
                                    ranked_num[self.rank] += 1
                                if number % 1000000 ==0 and self.rank==0:
                                    print('-------------------------------------------------------------- every 1M time:',(time.time()-star_time), '{}M'.format(number /1000000))
                    else:
                        link.barrier()
                        if self.rank != 0:
                            offset_file = None
                        if offset_file is not None:
                            print('Saving Offset File', offset_file, flush=True)
                            f_offset = open(offset_file, 'wb')
                        offset = 0
                        with open(each_meta_file) as f:
                            for line in f:
                                number += 1
                                random_rank = random.randint(0, self.world_size-1)
                                if offset_file is not None:
                                    f_offset.write(offset.to_bytes(6, byteorder='little'))
                                if self.rank == random_rank:
                                    ranked_line_offset[self.rank].append(offset)
                                    ranked_num[self.rank] += 1
                                offset += len(line.encode('UTF-8'))
                                #link.barrier()
                                if number % 1000000 ==0 and self.rank==0:
                                    print('-------------------------------------------------------------- every 1M time:',(time.time()-star_time), '{}M'.format(number /1000000))
                        if offset_file is not None:
                            f_offset.close()
                        #print('-------------------------------------------------------------------------',each_meta_file)
                    np.save(offset_ranked_file, ranked_line_offset[self.rank])
                if self.rank%8 == 0:
                    print('[Rank %d] Number of file'%self.rank, each_meta_file, len(ranked_line_offset[self.rank]),  flush=True)
                self.ranked_line_offsets[self.rank].append(ranked_line_offset[self.rank])


            self.line_offsets = self.ranked_line_offsets[self.rank]
            self.num = ranked_num[self.rank]

            # balance data length in each subprocess
            ranked_num = [torch.LongTensor([0]) for _ in range(self.world_size)]
            link.allgather(ranked_num, torch.LongTensor([self.num]))
            link.barrier()

            # import ipdb
            # ipdb.set_trace()
            max_num = max([item.item() for item in ranked_num])
            min_num = min([item.item() for item in ranked_num])
            if self.rank == 0:
                print(f'[Rank {self.rank}] Dataset Offset Loading Done! Ranked_num range is {min_num, max_num}', flush=True)
            if max_num > self.num:
                diff = max_num - self.num
                self.ranked_line_offsets[self.rank][0].extend(random.sample(self.ranked_line_offsets[self.rank][0], diff))
                self.num += diff
            else:
                assert self.num == max_num


        else:
            ranked_meta = [[] for _ in range(self.world_size)]
            ranked_num = [0 for _ in range(self.world_size)] 
            import time
            star_time = time.time()
            seed_num = 0
            for rd, each_meta_file in zip(root_dir, meta_files):
                with open(each_meta_file) as f:
                    #seed_num = 0
                    for line in f:
                        seed_num += 1
                        random.seed(seed_num)
                        random_rank = random.randint(0, self.world_size-1)
                        if self.rank == random_rank:
                            info = json.loads(line.encode('UTF-8'))
                            filename = joinpath(rd, info['filename'])
                            info['filename'] = filename

                            ranked_meta[self.rank].append(info)
                            ranked_num[self.rank] += 1
                        #link.barrier()
                        if seed_num % 1000000 ==0 and self.rank==0:
                            print('-------------------------------------------------------------- every 1M time:',(time.time()-star_time), '{}M'.format(seed_num /1000000))
                    #print('-------------------------',rd,ranked_num)

            self.metas = ranked_meta[self.rank]
            self.num = ranked_num[self.rank]

            # balance data length in each subprocess
            ranked_num = [torch.LongTensor([0]) for _ in range(self.world_size)]
            link.allgather(ranked_num, torch.LongTensor([self.num]))
            link.barrier()
            max_num = max([item.item() for item in ranked_num])
            if max_num > self.num:
                diff = max_num - self.num
                self.metas.extend(random.sample(self.metas, diff))
                self.num = len(self.metas)
            else:
                assert self.num == max_num

