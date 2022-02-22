try:
    from SpringCommonInterface import Metric as SCIMetric
except ImportError:
    SCIMetric = object


class Metric(SCIMetric):
    def __init__(self, metric={}, cmp_key=''):
        if SCIMetric != object:
            super(Metric, self).__init__(metric.get(cmp_key, -1))
        self.metric = metric
        self.cmp_key = cmp_key

    def set_cmp_key(self, key):
        self.cmp_key = key
        if SCIMetric != object:
            self.v = self.metric[self.cmp_key]


class Evaluator(object):
    def __init__(self):
        pass

    def eval(self, res_file):
        """
        This should return a dict with keys of metric names,
        values of metric values.

        Arguments:
            res_file (str): file that holds classification results
        """
        raise NotImplementedError

    @staticmethod
    def add_subparser(self, name, subparsers):
        raise NotImplementedError

    @staticmethod
    def from_args(cls, args):
        raise NotImplementedError
