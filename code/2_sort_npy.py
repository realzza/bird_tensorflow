import os
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import image
import random

def parse_args():
    desc="sort .png image files to .npy structure"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', type=str, default=None, help='/path/to/your/data/')
    parser.add_argument('--output', type=str, default=None, help='/path/to/output/dir/')
    parser.add_argument('--phase', type=str, default=None, help='train/val')
    return parser.parse_args()

args = parse_args()
data_dir = args.data
output_dir = args.output
phase = args.phase

all_birds = os.listdir(data_dir)
# bird to label
birds_dict = dict(zip(all_birds, range(len(all_birds))))

# collect all segs
all_segs = []
for bird in all_birds:
    all_segs += [data_dir + bird + '/' + x for x in os.listdir(data_dir+bird)]
print('... total: %d segments ...'%len(all_segs))

# shuffle all segments
random.shuffle(all_segs)

print('... start loading spectrograms ...')
data_arr = []
data_labels = []
for clip in tqdm(all_segs, desc="loading images"):
    data_arr.append(image.imread(clip))
    bird_name = clip.split('/')[-2]
    data_labels.append(birds_dict[bird_name])
data_arr = np.array(data_arr)
data_labels = np.array(data_labels)
print('... loading finished ...')

print('... start saving files ...')
# save as npy files
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
np.save(output_dir + 'X_%s'%phase, data_arr)
np.save(output_dir + 'y_%s'%phase, data_labels)
print('... saving completed ...')