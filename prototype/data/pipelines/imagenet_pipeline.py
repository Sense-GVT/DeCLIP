from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class ImageNetTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, crop, colorjitter=None, dali_cpu=False):
        super(ImageNetTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.data_input = ops.ExternalSource()
        self.label_input = ops.ExternalSource()
        self.colorjitter = colorjitter
        # let user decide which pipeline works him bets for RN version he runs
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB)
            self.res = ops.Resize(resize_x=crop, resize_y=crop)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to
            # handle all images from full-sized ImageNet without additional reallocations
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB,
                                            device_memory_padding=211025920, host_memory_padding=140544512)
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
        self.datas = self.data_input()
        self.labels = self.label_input()
        images = self.decode(self.datas)
        images = self.res(images)
        if self.colorjitter is not None:
            images = self.colorjit(images, brightness=self.rng_brightness(),
                                   contrast=self.rng_contrast(),
                                   saturation=self.rng_saturation(),
                                   hue=self.rng_hue())
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class ImageNetValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, crop, size):
        super(ImageNetValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.data_input = ops.ExternalSource()
        self.label_input = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.datas = self.data_input()
        self.labels = self.label_input()
        images = self.decode(self.datas)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
