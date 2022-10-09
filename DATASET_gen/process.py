from __future__ import absolute_import
import sys
import cv2
import math
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from random import random

sys.path.append('rainstreak_processor')
sys.path.append('raindrop_processor')
sys.path.append('hazy_processor')

from rainstreak_processor.rain_streak import rain_streak
from raindrop_processor.dropgenerator import generateDrops
from raindrop_processor.config import cfg
from hazy_processor.fog import *

def get_file_path(root_path):
    file_list, dir_list = [], []
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)
    return file_list, dir_or_files


def gen_img(index, img, ori_path, depth_path, stage1_path, stage2_path, stage3_path):
    global task_label
    haze_para = random()*90+60
    rain_para = random()*250+50
    raindrop_para = round(random()*17+5)

    cfg['maxDrops'] = cfg['minDrops'] = raindrop_para

    para = [img, haze_para, rain_para, raindrop_para]
    task_label.loc[index] = para
    # stage 1
    hz_img = hazy_gen(ori_path, depth_path, haze_para)
    cv2.imwrite(stage1_path, hz_img)

    # stage 2
    rs_gen = rain_streak(stage1_path, rain_para)
    rs_img = rs_gen.generate()
    cv2.imwrite(stage2_path, rs_img)

    # stage 3
    rd_gen = generateDrops(stage2_path, cfg)
    cv2.imwrite(stage3_path, rd_gen)


if __name__ == '__main__':
    global task_label
    task_label = pd.DataFrame(columns=['id', 'haze', 'rain', 'raindrop'])

    ori_root = r"C:/Users/95390/Desktop/DATASET/clean_image/"
    depth_root = r"C:/Users/95390/Desktop/DATASET/depth_image/"
    stage1_root = r"C:/Users/95390/Desktop/DATASET/degen_image/stage1/"
    stage2_root = r"C:/Users/95390/Desktop/DATASET/degen_image/stage2/"
    stage3_root = r"C:/Users/95390/Desktop/DATASET/degen_image/stage3/"

    ori_path_list, image_list = get_file_path(ori_root)
    for (index, img) in tqdm(enumerate(image_list)):
        ori_path = ori_root+img
        depth_path = depth_root+img.split('.')[0]+'.mat'
        stage1_path = stage1_root+img
        stage2_path = stage2_root+img
        stage3_path = stage3_root+img

        gen_img(index, img, ori_path, depth_path,
                stage1_path, stage2_path, stage3_path)
    task_label.to_csv(
        "C:/Users/95390/Desktop/DATASET/label/label.csv", index=0)
