import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from shutil import copyfile

# build the latin2eng dict
excel_dir = '/Netdata/2020/ziang/data/guangdong194/guangdong194_updated.xlsx'
bird_194 = pd.read_excel(excel_dir)
    
# create latin2eng dict
latin = list(bird_194['拉丁学名'])
eng = list(bird_194['英文名称'])
latin2eng = {''.join(eng[i].split()):'_'.join(latin[i].split()).lower() for i in range(len(latin))}

def parse_args():
    desc="sort full data into directories"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', type=str, default=None, help="path/to/data/")
    parser.add_argument('--output', type=str, default=None, help="path/to/outputdir/")
    parser.add_argument('--remove', type=bool, default=False, help='remove original file after soring or not')
    return parser.parse_args()

args=parse_args()
dataset = args.data
output = args.output
if not os.path.isdir(output):
    os.mkdir(output)
isremove = args.remove

all_clips = [dataset + x for x in os.listdir(dataset)]
print('... start sorting files ...')
for clip in tqdm(all_clips, desc="sorting full files into directories"):
    header = clip.split('/')[-1].split('_')[0]
    clip_name = '_'.join(clip.split('/')[-1].split('_')[1:])
    try:
        latin_name = latin2eng[header]
    except:
        latin_name = header
    if not os.path.isdir(output+latin_name):
        os.mkdir(output+latin_name)
    new_path = output + latin_name + '/' + clip_name
    copyfile(clip, new_path)
    if isremove:
        os.remove(clip)
print('... finished sorting ...')
print('... new directories at %s ...'%output)