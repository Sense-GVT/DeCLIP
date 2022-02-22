import json
import torch
import numpy as np
from .base_evaluator import Evaluator, Metric


class MultiClsMetric(Metric):
    def __init__(self, metric_dict={}):
        self.metric = metric_dict
        super(MultiClsMetric, self).__init__(self.metric)

    def __str__(self):
        return f'metric={self.metric} key={self.cmp_key}'

    def __repr__(self):
        return f'metric={self.metric} key={self.cmp_key}'


class MultiClsEvaluator(Evaluator):
    def __init__(self, topk=(1,)):
        super(MultiClsEvaluator, self).__init__()
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
        label = torch.from_numpy(np.array(res_dict['label_list']))
        label_name_list = res_dict['label_name_list'][0]
        res = {}
        num = pred.size(0)
        for i in range(label.size(1)):
            s_pred = pred[:, i, :]
            s_label = label[:, i]
            _, s_pred = s_pred.topk(1, 1, True, True)
            correct = s_pred.view(-1).eq(s_label)
            correct_k = correct.float().sum()
            acc = correct_k.mul_(100.0 / num)
            res.update({label_name_list[i]: acc.item()})

        metric = MultiClsMetric(res)
        return metric
