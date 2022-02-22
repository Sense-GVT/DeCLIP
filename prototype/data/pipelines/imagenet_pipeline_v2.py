# try:
#     import linklink.dali as link_dali  # noqa
# except ModuleNotFoundError:
#     print('import linklink.dali failed, linklink version should >= 0.2.0')

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class CustomPipeline(object):
    r"""CustomPipeline will work with :class:`linklink.dali.DataLoader` to
    provide pytorch native dataloader experience.

    """
    def __init__(self, **kwargs):
        self.extra_kwargs = kwargs

    def define_graph(self):
        raise NotImplementedError


class _PipelineBase(Pipeline):
    """
    Hide common options away.
    """
    def __init__(self, custom_pipe, batch_size, num_threads, device_id,
                 **kwargs):
        super(_PipelineBase, self).__init__(batch_size, num_threads, device_id,
                                            **kwargs)
        self.custom_pipe = custom_pipe

    def define_graph(self):
        return self.custom_pipe.define_graph()


class ImageNetTrainPipeV2(CustomPipeline):
    def __init__(self, data_root, data_list, sampler, crop, colorjitter=None):
        super(ImageNetTrainPipeV2, self).__init__()
        # print('data root: {}, data list: {}, len(sampler_index): {}'.format(
        #     data_root, data_list, len(sampler)))
        self.mc_input = ops.McReader(file_root=data_root,
                                     file_list=data_list,
                                     sampler_index=list(sampler))
        self.colorjitter = colorjitter

        dali_device = "gpu"
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all
        # images from full-sized ImageNet without additional reallocations
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                       device_memory_padding=211025920,
                                       host_memory_padding=140544512)

        self.res = ops.RandomResizedCrop(device=dali_device, size=(crop, crop))

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

        if self.colorjitter is not None:
            self.colorjit = ops.ColorTwist(device="gpu")
            self.rng_brightness = ops.Uniform(range=(1.0 - self.colorjitter[0], 1.0 + self.colorjitter[0]))
            self.rng_contrast = ops.Uniform(range=(1.0 - self.colorjitter[1], 1.0 + self.colorjitter[1]))
            self.rng_saturation = ops.Uniform(range=(1.0 - self.colorjitter[2], 1.0 + self.colorjitter[2]))
            self.rng_hue = ops.Uniform(range=(-self.colorjitter[3], self.colorjitter[3]))

    def define_graph(self):
        rng = self.coin()
        datas, labels = self.mc_input(name='Reader')
        images = self.decode(datas)
        images = self.res(images)
        if self.colorjitter is not None:
            images = self.colorjit(images, brightness=self.rng_brightness(),
                                   contrast=self.rng_contrast(),
                                   saturation=self.rng_saturation(),
                                   hue=self.rng_hue())
        output = self.cmnp(images, mirror=rng)
        return [output, labels]


class ImageNetValPipeV2(CustomPipeline):
    def __init__(self, data_root, data_list, sampler, crop, size):
        super(ImageNetValPipeV2, self).__init__()

        self.mc_input = ops.McReader(file_root=data_root,
                                     file_list=data_list,
                                     sampler_index=list(sampler))

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        datas, labels = self.mc_input()
        images = self.decode(datas)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, labels]
