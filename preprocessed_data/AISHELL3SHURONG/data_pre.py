# -*- coding: utf-8 -*-
# @Time    : 2021/5/5 下午3:40

import random

src_file_path = "/home/aone/lisen/code/FastSpeech2/preprocessed_data/AISHELL3SHURONG/shurong_train.txt"
tar_file_path = "/home/aone/lisen/code/FastSpeech2/preprocessed_data/AISHELL3SHURONG/shurong_half/train.txt"

src_lines = open(src_file_path).readlines()
random.shuffle(src_lines)

tmp_list = src_lines[:int(len(src_lines) / 2)] # 减半操作

with open(tar_file_path, 'a+') as f:
    for line in tmp_list:
        f.write(line)


print()

