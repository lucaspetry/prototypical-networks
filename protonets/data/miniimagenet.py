import os
import sys
import glob

from functools import partial

import numpy as np
import PIL
from PIL import Image

import torch
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import normalize, center_crop, to_tensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

MINIIMAGENET_DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data/miniimagenet')
MINIIMAGENET_CACHE = { }

def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d

def convert_tensor(key, d):
    d[key] = to_tensor(d[key])
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width), resample=PIL.Image.BILINEAR)
    return d

def normalize_image(key, stats, d):
    d[key] = normalize(d[key], mean=stats['mean'], std=stats['std'])
    return d

def load_class_images(d):
    label, rot = d['class'], -1

    if 'rot' in d['class']:
        label, rot = d['class'].split('/rot')
        rot = int(rot)

    if label not in MINIIMAGENET_CACHE:
        image_dir = os.path.join(MINIIMAGENET_DATA_DIR, 'data', label)

        class_images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        if len(class_images) == 0:
            raise Exception("No images found for miniimagenet class {} at {}.".format(label, image_dir))

        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             partial(scale_image, 'data', 84, 84),
                                             partial(convert_tensor, 'data')
                                             # partial(normalize_image, 'data', {'mean': (0.47234195 0.45386744 0.41036746),
                                             #                                   'std': (0.28678342 0.27806091 0.29304931)})
                                                                               ]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            MINIIMAGENET_CACHE[label] = sample['data']
            break # only need one sample because batch size equal to dataset length

    samples = MINIIMAGENET_CACHE[label]

    # Rotates images if needed
    if rot != -1:
        nRot = rot // 90
        samples = torch.rot90(samples.cuda(), nRot, dims=[2, 3]).cpu()

    return { 'class': d['class'], 'data': samples }

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    split_dir = os.path.join(MINIIMAGENET_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        transforms = [partial(convert_dict, 'class'),
                      load_class_images,
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = []
        with open(os.path.join(split_dir, "{:s}.csv".format(split)), 'r') as f:
            for class_name in f.readlines():
                name = class_name.split(',')[1].rstrip('\n')

                if name == 'label':
                    continue
                
                if opt['data.augmented']:
                    class_names.extend([name + '/rot000', name + '/rot090',
                                        name + '/rot180', name + '/rot270'])
                else:
                    class_names.append(name)
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
