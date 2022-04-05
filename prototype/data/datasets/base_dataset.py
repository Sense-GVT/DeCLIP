import os
import linklink as link
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import mc
except ImportError:
    pass
# import ceph
# from petrel_client.client import Client


class BaseDataset(Dataset):
    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None,
                 read_from='mc',
                 evaluator=None):

        super(BaseDataset, self).__init__()

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.read_from = read_from
        self.evaluator = evaluator
        self.initialized = False
        
        if self.read_from == 'petrel':
            self._init_petrel()
        if self.read_from == 'petrel_bigmodel':
            self._init_petrel('~/petreloss.conf')
        if self.read_from == 'petrel_bigmodel_mc':
            self._init_petrel('~/petreloss.conf.mc')

    def __len__(self):
        """
        Returns dataset length
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Get a single image data: from dataset

        Arguments:
            - idx (:obj:`int`): index of image, 0 <= idx < len(self)
        """
        raise NotImplementedError

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _init_ceph(self):
        if not self.initialized:
            # self.s3_client = ceph.S3Client('~/.s3cfg')
            self.initialized = True

    def _init_petrel(self, path=None):
        if not self.initialized:
            from petrel_client.client import Client
            if path is None:
                self.client = Client('/mnt/lustre/zhaolichen/codes/penghuan/clip_stage1/petreloss.conf')
            else:
                self.client = Client(path)
            self.initialized = True

    def _init_osg(self):
        if not self.initialized:
            from spring_sdk import OSG
            self.osg_client = OSG(self.osg_server, secure=False)
            self.initialized = True

    def read_file(self, meta_dict):
        if self.read_from == 'fake':
            if self.initialized:
                filebytes = self.saved_filebytes
            else:
                filebytes = self.saved_filebytes = np.fromfile(meta_dict['filename'], dtype=np.uint8)
                self.initialized = True
        elif self.read_from == 'mc':
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(meta_dict['filename'], value)
            value_str = mc.ConvertBuffer(value)
            filebytes = np.frombuffer(value_str.tobytes(), dtype=np.uint8)
        elif self.read_from == 'ceph':
            self._init_ceph()
            value = self.s3_client.Get(meta_dict['filename'])
            filebytes = np.frombuffer(value, dtype=np.uint8)
        elif self.read_from in ['petrel', 'petrel_bigmodel', 'petrel_bigmodel_mc']:
            self._init_petrel()
            value = self.client.Get(meta_dict['filename'])
            filebytes = np.frombuffer(value, dtype=np.uint8)
        elif self.read_from == 'fs':
            filebytes = np.fromfile(meta_dict['filename'], dtype=np.uint8)
        elif self.read_from == 'osg':
            self._init_osg()
            img_str = self.osg_client.get_object(meta_dict['bucket'], meta_dict['key'])
            filebytes = np.fromstring(img_str, np.uint8)
        else:
            raise RuntimeError("unknown value for read_from: {}".format(self.read_from))

        return filebytes

    def dump(self, writer, output):
        """
        Dump classification results

        Arguments:
            - writer: output stream
            - output (:obj:`dict`): different for imagenet and custom
        """
        raise NotImplementedError

    def merge(self, prefix):
        """
        Merge results into one file.

        Arguments:
            - prefix (:obj:`str`): dir/results.rank
        """
        world_size = link.get_world_size()
        merged_file = prefix.rsplit('.', 1)[0] + '.all'
        merged_fd = open(merged_file, 'w')
        for rank in range(world_size):
            res_file = prefix + str(rank)
            assert os.path.exists(res_file), f'No such file or directory: {res_file}'
            with open(res_file, 'r') as fin:
                for line_idx, line in enumerate(fin):
                    merged_fd.write(line)
        merged_fd.close()
        return merged_file

    def inference(self, res_file):
        """
        Arguments:
            - res_file (:obj:`str`): filename of result
        """
        prefix = res_file.rstrip('0123456789')
        merged_res_file = self.merge(prefix)
        return merged_res_file

    def evaluate(self, res_file):
        """
        Arguments:
            - res_file (:obj:`str`): filename of result
        """
        prefix = res_file.rstrip('0123456789')
        merged_res_file = self.merge(prefix)
        metrics = self.evaluator.eval(merged_res_file) if self.evaluator else {}
        return metrics

    def tensor2numpy(self, x):
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.cpu().numpy()
        if isinstance(x, list):
            x = [_.cpu().numpy() if torch.is_tensor(_) else _ for _ in x]
        return x
