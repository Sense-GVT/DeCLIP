import os
import cv2
import json
import numpy as np


def generate_nnie_config(nnie_cfg, config, nnie_out_path='./config.json', tensor_type='float'):
    """
    Generate NNIE config for spring.nart.switch, details in
    http://spring.sensetime.com/docs/nart/tutorial/switch/nnie.html
    """
    u8_start = False if tensor_type == 'float' else False
    default_config = {
        "default_net_type_token": "nnie",
        "rand_input": False,
        "data_num": 100,
        "input_path_map": {
            "data": "./image_bins",
        },
        "nnie": {
            "max_batch": 1,
            "output_names": [],
            "mapper_version": 11,
            "u8_start": u8_start,
            "device": "gpu",
            "verbose": False,
            "image_path_list": ["./image_list.txt"],
            "mean": [128, 128, 128],
            "std": [1, 1, 1]
        }
    }
    image_path_list = nnie_cfg['image_path_list']
    assert os.path.exists(image_path_list)
    with open(image_path_list, 'r') as f:
        image_list = [item.strip() for item in f.readlines()]

    mean = config.to_kestrel.get('pixel_means', [123.675, 116.28, 103.53])
    std = config.to_kestrel.get('pixel_stds', [58.395, 57.12, 57.375])
    resize_hw = config.to_kestrel.get('resize_hw', (224, 224))
    resize_hw = tuple(resize_hw)
    data_num = len(image_list)
    image_bin_path = generate_image_bins(image_list, mean, std, resize_hw)
    default_config['data_num'] = data_num
    default_config['input_path_map']['data'] = image_bin_path
    default_config['nnie']['max_batch'] = nnie_cfg.get('max_batch', 1)
    default_config['nnie']['mapper_version'] = nnie_cfg.get('mapper_version', 11)
    default_config['nnie']['image_path_list'] = [image_path_list]
    default_config['nnie']['mean'] = [128] * len(std)
    default_config['nnie']['std'] = [1] * len(std)
    with open(nnie_out_path, "w") as f:
        json.dump(default_config, f, indent=2)

    return nnie_out_path


def generate_image_bins(image_list, mean, std, resize_hw, image_bins_folder='./image_bins'):
    """
    Generate data for calibration.
    """
    if not os.path.exists(image_bins_folder):
        os.makedirs(image_bins_folder)
    else:
        os.system("rm -r {}/*".format(image_bins_folder))

    for i, image in enumerate(image_list):
        output_bin = os.path.join(image_bins_folder, str(i) + ".bin")
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, resize_hw)
        if len(mean) == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img, dtype=np.float32)
            img = (img - mean[0]) / std[0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.float32)
            for i in range(len(mean)):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
            img = img.transpose((2, 0, 1))
        bin_str = img.astype('f').tostring()
        with open(output_bin, 'wb') as fp:
            fp.write(bin_str)
    return image_bins_folder
