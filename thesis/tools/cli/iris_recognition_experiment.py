import streamlit as st

import click

import os
from collections import defaultdict
import json
import yaml
import numpy as np
import random
import cv2 as cv
from tqdm import tqdm

from multiprocessing import Pool

import seaborn as sns

from thesis.segmentation import IrisImage, IrisSegmentation, SKImageIrisCodeEncoder
from thesis.data import SegmentationDataset

# base = '/home/anton/data/eyedata/iris'
base = '/Users/Anton/Desktop/data/iris'


def create_code(encoder, sample, angles=1, angular_spacing=5):
    if angles == 1:
        ic = encoder.encode(sample)
        return [ic]
    else:
        angular_spacing_radians = angular_spacing / 360 * 2 * np.pi
        codes = []
        for a in np.linspace(-angular_spacing_radians / 2 * angles, angular_spacing_radians / 2 * angles,
                             angles):
            codes.append(encoder.encode(sample, start_angle=a))
        return codes


@click.command()
@click.argument('config')
def main(config):
    with open(os.path.join('configs/iris_recognition', f'{config}.yaml')) as file:
        config = yaml.safe_load(file)

        dataset_path = config['dataset']
        print('[INFO] Loading dataset')
        dataset = SegmentationDataset.from_path(dataset_path)
        print('[INFO] Data loaded')

        parameters = config['parameters']
        scales = parameters['scales']
        angles = parameters['angles']
        angular = parameters['resolution']['angular']
        radial = parameters['resolution']['radial']
        eps = parameters['epsilon']

        rotation = parameters['rotation']
        num_rotations = rotation['num']
        step_size = rotation['step_size']

        encoder = SKImageIrisCodeEncoder(angles, angular, radial, scales, eps)

        pool = Pool()

        codes = []
        pool.imap()
        for item in tqdm(dataset.samples):
            codes.append(create_code(item.image, num_rotations, step_size))

        n = len(codes)
        distance_matrix = np.zeros((n, n))
        same_mask = np.zeros((n, n), np.bool)
        num_samples = 0
        for i in tqdm(range(n), total=n):
            for j in range(n):
                in_data = dataset.samples
                info_i = in_data[i]
                info_j = in_data[j]
                same = False
                if info_i.user_id == info_j.user_id and info_i.eye == info_j.eye:
                    same = True
                    same_mask[i, j] = True

                if same or random.random() < 2:  # Rate
                    num_samples += 1
                    distance_matrix[i, j] = min([codes[i][num_rotations // 2].dist(cb) for cb in codes[j]])

        intra_distances = []
        inter_distances = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if same_mask[i, j]:
                    intra_distances.append(distance_matrix[i, j])
                else:
                    inter_distances.append(distance_matrix[i, j])

        intra_distances = np.array(intra_distances)
        inter_distances = np.array(inter_distances)

        with open(os.path.join('results', 'recognition', f'{config}.json'), 'w') as file:
            json.dump({
                'config': config,
                'results': {
                    'intra_distances': list(intra_distances),
                    'inter_distances': list(inter_distances),
                }
            }, file)


if __name__ == '__main__':
    main()
