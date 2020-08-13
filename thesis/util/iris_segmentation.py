import os
from typing import Dict, List

import cv2 as cv
import numpy as np
from enum import Enum
from dataclasses import dataclass
from glob2 import glob

from tqdm import tqdm
import json
import click


class Dataset(Enum):
    CASIA = 'casia4i'
    IITD = 'iitd'
    UBIRIS = 'ubiris'


@dataclass
class IrisImage:
    image_path: str
    points: Dict[str, List[List[float]]]


def read_points_from_file(path):
    with open(path) as f:
        lines = f.readlines()
        tokens = [line.split() for line in lines]
        coords = [[float(t) for t in line] for line in tokens]
        return coords


def get_segmented_images(dataset: str, path: str):
    image_folder = os.path.join(path, 'images', dataset)
    segment_folder = os.path.join(path, 'segmentation/IRISSEG-EP/dataset', dataset)

    exts = ('.jpg', '.png', '.tiff')
    images = set()
    for ext in exts:
        images |= set(glob(os.path.join(image_folder, f'**/**{ext}'), recursive=True))
    segmentations = set(glob(os.path.join(segment_folder, '**/**.txt'), recursive=True))

    res = []

    for img_path in tqdm(images):
        # img = cv.imread(img_path)

        img_base = os.path.basename(img_path)
        img_name = os.path.splitext(img_base)[0]

        files = filter(lambda x: img_name in x, segmentations)
        points = {
            os.path.basename(f).split(os.extsep)[1]: read_points_from_file(f)
            for f in files
        }

        res.append({
            'image': os.path.abspath(img_path),
            'points': points
        })

    return {
        'dataset': dataset,
        'data': res
    }


@click.group()
def data():
    pass


@data.group()
def iris():
    pass


@iris.command()
@click.argument('path')
@click.argument('dataset')
@click.argument('output')
def create(path, dataset, output):
    '''Create json file containing dataset points and image file paths.

    PATH is the base path to the "iris" dataset.

    DATASET is the subfolder in the "images" folder to create a json file for.

    OUTPUT is the path to the resulting json file.
    '''
    dset = get_segmented_images(dataset, path)
    with open(output, 'w') as f:
        json.dump(dset, f)


if __name__ == '__main__':
    data()