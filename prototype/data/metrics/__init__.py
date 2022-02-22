from .imagenet_evaluator import ImageNetEvaluator
from .custom_evaluator import CustomEvaluator
from .multiclass_evaluator import MultiClsEvaluator


def build_evaluator(cfg):
    evaluator = {
        'custom': CustomEvaluator,
        'imagenet': ImageNetEvaluator,
        'multiclass': MultiClsEvaluator,
    }[cfg['type']]
    return evaluator(**cfg['kwargs'])
