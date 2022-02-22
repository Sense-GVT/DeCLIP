import json
import yaml
import torch
import numpy as np
from .base_evaluator import Evaluator, Metric


class ClsMetric(Metric):
    def __init__(self, metric_dict={}):
        self.metric = metric_dict
        super(ClsMetric, self).__init__(self.metric)

    def __str__(self):
        return f'metric={self.metric} key={self.cmp_key}'

    def __repr__(self):
        return f'metric={self.metric} key={self.cmp_key}'

    def set_cmp_key(self, key):
        self.cmp_key = key
        self.v = self.metric[self.cmp_key]


class ImageNetEvaluator(Evaluator):
    def __init__(self, topk=[1, 5]):
        super(ImageNetEvaluator, self).__init__()
        self.topk = topk

    def load_res(self, res_file):
        """
        Load results from file.
        """
        res_dict = {}
        with open(res_file) as f:
            lines = f.readlines()
        for line in lines:
            info = json.loads(line)
            for key in info.keys():
                if key not in res_dict.keys():
                    res_dict[key] = [info[key]]
                else:
                    res_dict[key].append(info[key])
        return res_dict

    def eval(self, res_file):
        res_dict = self.load_res(res_file)
        pred = torch.from_numpy(np.array(res_dict['score']))
        label = torch.from_numpy(np.array(res_dict['label']))
        num = pred.size(0)
        maxk = max(self.topk)
        _, pred = pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))
        res = {}
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / num)
            res.update({f'top{k}': acc.item()})
        metric = ClsMetric(res)
        metric.set_cmp_key(f'top{self.topk[0]}')

        return metric

    @staticmethod
    def add_subparser(name, subparsers):
        subparser = subparsers.add_parser(
            name, help='subcommand for ImageNet of Top-1/5 accuracy metric')
        subparser.add_argument('--config', dest='config', required=True,
                               help='settings of classification in yaml format')
        subparser.add_argument('--res_file', required=True, action='append',
                               help='results file of classification')

        return subparser

    @classmethod
    def from_args(cls, args):
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        kwargs = config['data']['evaluator']['kwargs']

        return cls(**kwargs)
