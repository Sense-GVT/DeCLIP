from .clip import (  # noqa: F401
    clip_res50, clip_vitb32
)

from .declip import declip_res50, declip_vitb32

from .filip import filip_res50, filip_vitb32

from .slip import slip_res50, slip_vitb32

from .defilip import defilip_vitb32



def model_entry(config):

    if config['type'] not in globals():
        from prototype.spring import PrototypeHelper
        return PrototypeHelper.external_model_builder[config['type']](**config['kwargs'])

    return globals()[config['type']](**config['kwargs'])
