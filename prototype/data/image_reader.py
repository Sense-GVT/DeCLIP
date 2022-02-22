import io
from PIL import Image
import logging
import kestrel as ks
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger('global')



def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return img


def kestrel_loader(img_bytes, filepath):
    input_frame = ks.Frame()
    try:
        image_data = img_bytes.tobytes()
        input_frame.create_from_mem(image_data, len(image_data))
        if input_frame.frame_type != ks.KESTREL_VIDEO_RGB:
            input_frame = input_frame.cvt_color(ks.KESTREL_VIDEO_RGB)
        if ks.Device().mem_type() == ks.KESTREL_MEM_DEVICE:
            input_frame = input_frame.upload()
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return [input_frame]


def build_image_reader(reader_type):
    if reader_type == 'pil':
        return pil_loader
    elif reader_type == 'kestrel':
        return kestrel_loader
    else:
        raise NotImplementedError
