import enum
import os
import re
import random
from abc import ABC, abstractmethod
from typing import Dict, List

import cv2 as cv
import numpy as np
from enum import Enum
from dataclasses import dataclass
from glob2 import glob

from tqdm import tqdm
import json
import click

from thesis.segmentation import IrisSegmentation, IrisImage, IrisCode


class EyeSide(str, Enum):
    LEFT = 'left'
    RIGHT = 'right'


@dataclass
class ImageInfo:
    user_id: str
    eye: EyeSide
    image_id: str
    session_id: str


class DataFormat(ABC):
    @staticmethod
    @abstractmethod
    def get_info(name) -> ImageInfo:
        ...


class UbirisFormat(DataFormat):

    @staticmethod
    def get_info(name) -> ImageInfo:
        matches = re.match(r'C(?P<user_id>[0-9]+)_S(?P<session_id>[0-9]+)_I(?P<image_id>[0-9]+)', name)
        # This conversion is done to preserve IDs but keep ID equality between left and right eyes
        user_id = str(int(matches.group('user_id')) // 2 * 2)
        eye = EyeSide.RIGHT if int(matches.group('user_id')) % 2 else EyeSide.LEFT

        return ImageInfo(
            user_id, eye,
            image_id=matches.group('image_id'),
            session_id=matches.group('session_id'))


class CasiaIVFormat(DataFormat):
    @staticmethod
    def get_info(name) -> ImageInfo:
        matches = re.match(r'S(?P<user_id>[0-9]+)(?P<eye_side>L|R)(?P<image_id>[0-9]+)', name)
        # This conversion is done to preserve IDs but keep ID equality between left and right eyes
        eye = EyeSide.RIGHT if matches.group('eye_side') == 'R' else EyeSide.LEFT
        return ImageInfo(
            user_id=matches.group('user_id'),
            eye=eye,
            image_id=matches.group('image_id'),
            session_id='')


def read_points_from_file(path):
    with open(path) as f:
        lines = f.readlines()
        tokens = [line.split() for line in lines]
        coords = [[float(t) for t in line] for line in tokens]
        return coords


def get_segmented_images(dataset: str, path: str):
    image_folder = os.path.join(path, 'images', dataset)
    segment_folder = os.path.join(path, 'segmentation/IRISSEG-EP/dataset', dataset)

    extensions = ('.jpg', '.png', '.tiff', '.bmp')
    images = set()
    for ext in extensions:
        images |= set(glob(os.path.join(image_folder, f'**/**{ext}'), recursive=True))
    segmentations = set(glob(os.path.join(segment_folder, '**/**.txt'), recursive=True))

    res = []

    formatter = None
    if 'casia4i' in dataset:
        formatter = CasiaIVFormat()
    elif dataset == 'ubiris':
        formatter = UbirisFormat()

    for img_path in tqdm(images):
        img_base = os.path.basename(img_path)
        img_name = os.path.splitext(img_base)[0]

        files = filter(lambda x: img_name in x, segmentations)
        points = {
            os.path.basename(f).split(os.extsep)[1]: read_points_from_file(f)
            for f in files
        }

        if len(points) == 0:  # Only append files that have corresponding annotations
            continue

        res.append({
            'image': os.path.abspath(img_path),
            'points': points,
            'info': formatter.get_info(img_name).__dict__
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
@click.option('-l', '--limit', type=int, default=0)
def create(path, dataset, output, limit):
    """Create json file containing dataset points and image file paths.

    PATH is the base path to the "iris" dataset.

    DATASET is the subfolder in the "images" folder to create a json file for.

    OUTPUT is the path to the resulting json file.
    """
    dset = get_segmented_images(dataset, path)
    if limit > 0:
        dset['data'] = random.sample(dset['data'], limit)
    with open(output, 'w') as f:
        json.dump(dset, f)


@data.command()
@click.argument('path')
def preview(path):
    """Preview image and mask.

    PATH is a path to a json dataset."""
    with open(path) as f:
        data = json.load(f)['data']
        while True:
            index = click.prompt(f'Type an index to view (range 0 to {len(data)})', type=int)
            if index >= len(data) or index < 0:
                print(f'Invalid index!')
                continue
            point = data[index]
            img = cv.imread(point['image'])
            seg = IrisSegmentation.from_dict(point['points'])
            mask = seg.get_mask((img.shape[1], img.shape[0]))
            iris_img = IrisImage(seg, img)

            masked = cv.bitwise_and(img, img, mask=iris_img.mask * 255)
            cv.imshow('Preview', masked)

            polar, mask = iris_img.to_polar(200, 100)
            cv.imshow('Polar', polar)
            masked = cv.bitwise_and(polar, polar, mask=mask * 255)
            cv.imshow('PolarMask', masked)

            code = IrisCode(iris_img)

            code_img = np.array(code.code).reshape((20, -1))
            print(code.code)

            cv.imshow('Code', code_img)

            cv.waitKey(100)


@data.group()
def track():
    pass


def combine_json(glints: list, pupils: list, positions: list, calibration_samples: int) -> dict:
    # if len(glints) != len(pupils) != len(positions):
    #     raise ValueError("Data files contain different numbers of elements.")

    # combined = [{
    #     'image': f'{i}.jpg',
    #     'glints': img_glints,
    #     'pupil': pupil,
    #     'position': position
    # } for i, (img_glints, pupil, position) in enumerate(zip(glints, pupils, positions))]
    combined = [{
        'image': f'{i}.jpg',
        'position': position
    } for i, position in enumerate(positions)]

    return {
        'calibration': combined[:calibration_samples],
        'test': combined[calibration_samples:]
    }


@track.command()
@click.argument('path')
@click.argument('calibration_samples', type=int)
def create(path: str, calibration_samples: int):
    """Create single json file.

    PATH: Path to folder containing images and json files.
    OUTPUT: Filename of output json file.
    CALIBRATION_SAMPLES: Number of images used for calibration."""

    glints_path = os.path.join(path, 'glints.json')
    pupils_path = os.path.join(path, 'pupils.json')
    positions_path = os.path.join(path, 'positions.json')
    # open(glints_path) as glints_file, open(pupils_path) as pupils_file,
    with open(positions_path) as positions_file:
        glints = None
        pupils = None
        # glints = json.load(glints_file)
        # pupils = json.load(pupils_file)
        positions = json.load(positions_file)


        output_data = combine_json(glints, pupils, positions, calibration_samples)

        with open(os.path.join(path, 'data.json'), 'w') as output_file:
            json.dump(output_data, output_file, indent=4)

        print("Created data.json")


if __name__ == '__main__':
    data()
