import streamlit as st

import click

import copy
import os
from collections import defaultdict
import json
import yaml
import numpy as np
import random
import cv2 as cv
from tqdm import tqdm

from multiprocessing import Pool
from itertools import repeat, product

import seaborn as sns

from thesis.segmentation import IrisImage, IrisSegmentation, SKImageIrisCodeEncoder
from thesis.data import SegmentationDataset
from thesis.optim import filters

# base = '/home/anton/data/eyedata/iris'
base = '/Users/Anton/Desktop/data/iris'


def create_code(args):
    encoder, sample, angles, angular_spacing = args
    sample = sample.image
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


def compare(args):
    (left, right), codes_left, codes_right, (i, j), num_rotations = args

    same = False
    if left.user_id == right.user_id and left.eye == right.eye:
        same = True

    distance = min([codes_left[i][num_rotations // 2].dist(cb) for cb in codes_right[j]])

    return (i, j), (distance, same)


# def create_row(args):
#     dataset, codes, i, n, num_rotations = args
#
#     distance_matrix = np.zeros((1, n))
#     same_mask = np.zeros((1, n), np.bool)
#
#     for j in range(n):
#         in_data = dataset.samples
#         info_i = in_data[i]
#         info_j = in_data[j]
#         same = False
#         if info_i.user_id == info_j.user_id and info_i.eye == info_j.eye:
#             same = True
#             same_mask[0, j] = True
#
#         if same or random.random() < 2:  # Rate
#             distance_matrix[0, j] = min([codes[i][num_rotations // 2].dist(cb) for cb in codes[j]])
#
#     return distance_matrix, same_mask


@click.command()
@click.argument('config')
@click.argument('output')
@click.option('--filter', help='Apply filter configuration')
def main(config, output, filter):
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

        args = list(zip(repeat(encoder), dataset.samples, repeat(num_rotations), repeat(step_size)))

        # codes = []
        # pool = Pool(processes=7)
        codes = list(tqdm(map(create_code, args), total=len(dataset)))
        # for item in tqdm(dataset.samples):
        #     codes.append(create_code(encoder, item.image, num_rotations, step_size))
        print('[INFO] codes created')
        n = len(codes)

        comparisons = list(product(range(n), range(n)))
        comparison_samples = map(lambda v: (dataset.samples[v[0]], dataset.samples[v[1]]), comparisons)

        if filter is not None:
            print('[INFO] Applying filter and creating filtered codes')
            args = config['filters'][filter]
            filter_func = getattr(filters, filter)

            filter_samples = copy.deepcopy(dataset.samples)

            for s in tqdm(filter_samples):
                s.image.image = filter_func(s.image.image, **args)

            print('[INFO] creating codes')
            args = list(zip(repeat(encoder), filter_samples, repeat(num_rotations), repeat(step_size)))
            f_codes = list(tqdm(map(create_code, args), total=len(dataset)))
            print('[INFO] codes created')
            n = len(f_codes)

            args = list(zip(comparison_samples, repeat(codes), repeat(f_codes), comparisons, repeat(num_rotations)))
        else:
            args = list(zip(comparison_samples, repeat(codes), repeat(codes), comparisons, repeat(num_rotations)))
        # args = list(zip(repeat(dataset), repeat(codes), range(n), repeat(n), repeat(num_rotations)))

        print('[INFO] comparing codes')

        # results = list(tqdm(map(create_row, args), n))
        results = list(tqdm(map(compare, args), total=len(args)))
        distance_matrix = np.zeros((n, n))
        same_mask = np.zeros((n, n), np.bool)
        for (i, j), (distance, same) in results:
            distance_matrix[i, j] = distance
            same_mask[i, j] = same

        # pool.close()
        # pool.join()
        # distances, masks = zip(*results)

        # distance_matrix = np.vstack(distances)
        # same_mask = np.vstack(masks)

        # for i in tqdm(range(n), total=n):
        #     for j in range(n):
        #         in_data = dataset.samples
        #         info_i = in_data[i]
        #         info_j = in_data[j]
        #         same = False
        #         if info_i.user_id == info_j.user_id and info_i.eye == info_j.eye:
        #             same = True
        #             same_mask[i, j] = True
        #
        #         if same or random.random() < 2:  # Rate
        #             num_samples += 1
        #             distance_matrix[i, j] = min([codes[i][num_rotations // 2].dist(cb) for cb in codes[j]])

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

        with open(os.path.join('results', 'recognition', f'{output}.json'), 'w') as file:
            json.dump({
                'config': config,
                'results': {
                    'intra_distances': list(intra_distances),
                    'inter_distances': list(inter_distances),
                }
            }, file)


if __name__ == '__main__':
    main()
