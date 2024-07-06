#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# Created on 2024-02-01
# @Author: gy
# path: /home/gy/MyWork/stroke.py
# run: python3.7 stroke
# result: (74.8656, 4, 560) (avg, min, max)

# 统计FSCOCO数据集中，每张草图最多、最少、平均笔画数。

import os
import json
from tqdm import tqdm

def get_files_in_folders(path):
    # 获取指定路径下的所有文件夹
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    # 遍历每个文件夹，获取其中的文件
    all_files = []
    for folder in folders:
        folder_path = os.path.join(path, folder)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        all_files.extend(files)

    return all_files

def count(files):
    total = 0
    max = 0
    min = float('inf')
    for i in tqdm(files):
        # 读取JSON文件
        # print("file: ", i)
        with open(i, 'r') as file:
            # 加载JSON数据
            data = json.load(file)
            # 统计stroke数量
            num = sum(1 for item in data if item.get('pen_state') == [1, 0, 0])
            max = num if num > max else max
            min = num if num < max else min
            total += num

    avg = total / len(files)

    return avg, min, max

# 指定路径
path_to_search = "/home/gy/datasets/FScoco/raw_data"

# 获取所有文件路径
files = get_files_in_folders(path_to_search)
# 打印结果
print(count(files))
