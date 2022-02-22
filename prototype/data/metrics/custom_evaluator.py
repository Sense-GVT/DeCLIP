import math
import json
import numpy as np
from sklearn import metrics
from .base_evaluator import Evaluator, Metric
from prototype.utils.misc import get_logger


class CustomMetric(Metric):
    def __init__(self, metric_dict={}):
        self.metric = metric_dict
        super(CustomMetric, self).__init__(self.metric)

    def __str__(self):
        return f'metric={self.metric} key={self.cmp_key}'

    def __repr__(self):
        return f'metric={self.metric} key={self.cmp_key}'

    def set_cmp_key(self, key):
        self.cmp_key = key
        # TODO fix key_name
        # self.v = self.metric[self.cmp_key]


class CustomEvaluator(Evaluator):
    def __init__(self,
                 key_metric,
                 defect_classes,
                 recall_thres,
                 tpr_thres):
        super(CustomEvaluator, self).__init__()
        self.key_metric = key_metric
        self.recall_thres = recall_thres
        self.tpr_thres = tpr_thres
        self.defect_classes = defect_classes
        self.logger = get_logger(__name__)

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

    def calculate_recall_tpr(self, sorted_labels):
        total_defects = sum(sorted_labels)
        total_norm = sum(1 - sorted_labels)

        defect_sums = np.cumsum(sorted_labels)
        norm_sums = np.cumsum(1 - sorted_labels)

        tpr_recall_dic = {}
        for current_num, _ in enumerate(sorted_labels[:-1]):
            current_defect = defect_sums[current_num]
            current_recall = float(current_defect) / total_defects

            current_norm = total_norm - norm_sums[current_num + 1]
            current_tpr = float(current_norm) / total_norm
            for one_tpr_thres in self.tpr_thres:
                if current_tpr >= one_tpr_thres:
                    tpr_recall_dic[one_tpr_thres] = max(tpr_recall_dic.get(one_tpr_thres, -np.inf), current_recall)
        ret_metrics = {}
        # recall@tpr
        for (one_tpr, one_recall) in tpr_recall_dic.items():
            ret_metrics['recall@tpr{}'.format(one_tpr)] = one_recall
        return ret_metrics

    def calculate_recall_fpr(self, sorted_labels, sorted_probs):

        total_pos_nums = sum(sorted_labels)
        total_neg_nums = len(sorted_labels) - total_pos_nums

        target_recall_nums = [math.ceil(i * total_pos_nums) for i in self.recall_thres]
        self.logger.info('recall thres: {}'.format(self.recall_thres))
        self.logger.info('target recall nums: {}'.format(target_recall_nums))
        recall_precsion_vecs = []
        fnr_fpr_vecs = []
        for target_num in target_recall_nums:
            recall_nums = 0
            fp = 0
            for i in range(0, len(sorted_labels)):
                # defect_classes = 1
                if sorted_labels[i] == 1:
                    recall_nums += 1
                else:
                    fp += 1
                recall = recall_nums / total_pos_nums
                prec = recall_nums / (i + 1)
                score = sorted_probs[i]
                fnr = 1 - recall
                fpr = fp / total_neg_nums

                if recall_nums == target_num:
                    recall_precsion_vecs.append((recall, prec, score))
                    fnr_fpr_vecs.append((fnr, fpr))
                    break
        ret_metrics = {}
        for i in range(len(recall_precsion_vecs)):
            fnr = (1 - self.recall_thres[i]) * 100
            fnr = round(fnr)
            # fpr@recall
            ret_metrics['fpr@{}recall'.format(recall_precsion_vecs[i][0])] = fnr_fpr_vecs[i][1]

            self.logger.info(
                "recall: {:.5f}, fpr: {:.5f}, score: {:.5f}".format(
                    recall_precsion_vecs[i][0],
                    fnr_fpr_vecs[i][1],
                    recall_precsion_vecs[i][2]))

        return ret_metrics

    def performances(self, labels, scores):
        ret_metrics = {}

        scores = np.asarray(scores)
        # sum all the pos probs, use one thres
        all_pos_probs = scores[:, self.defect_classes].sum(axis=1)
        neg_pos_labels = np.asarray(labels, dtype=int)
        neg_pos_labels = list(map(lambda x: 1 if x in self.defect_classes else 0, neg_pos_labels))
        neg_pos_labels = np.asarray(neg_pos_labels, dtype=int)

        sorted_idx = np.argsort(all_pos_probs)[::-1]
        sorted_labels = neg_pos_labels[sorted_idx]
        sorted_probs = all_pos_probs[sorted_idx]

        # auc
        fpr, tpr, _ = metrics.roc_curve(neg_pos_labels, all_pos_probs, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        ret_metrics['auc'] = auc
        # ap
        ap = metrics.average_precision_score(neg_pos_labels, all_pos_probs)
        ret_metrics["AP"] = ap
        # fpr recall
        ret_metrics.update(self.calculate_recall_fpr(sorted_labels, sorted_probs))
        # tpr recall
        ret_metrics.update(self.calculate_recall_tpr(sorted_labels))

        return ret_metrics

    def eval(self, res_file):
        res_dict = self.load_res(res_file)
        labels = res_dict['label']
        scores = res_dict['score']
        ret_metrics = self.performances(labels, scores)
        metric = CustomMetric(ret_metrics)
        metric.set_cmp_key(self.key_metric)

        return metric
