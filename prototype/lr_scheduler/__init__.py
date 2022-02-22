from .scheduler import Step, StepDecay, Cosine, Poly # noqa F401


def epoch_to_iter(config):
    ratio = config.max_iter / config.max_epoch
    candidate_key = list(config.keys())
    for k in candidate_key:
        if k == 'lr_epochs':
            config['lr_steps'] = [round(epoch*ratio) for epoch in config[k]]
        elif k == 'warmup_epoch':
            config['warmup_steps'] = max(round(config[k]*ratio), 2)
        else:
            continue
        config.pop(k)
    config.pop('max_epoch')


def scheduler_entry(config):
    if config.type in ['StepEpoch', 'CosineEpoch']:
        config.type = config.type.replace('Epoch', '')
        epoch_to_iter(config.kwargs)
    return globals()[config.type](**config.kwargs)
