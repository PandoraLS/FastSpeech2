# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 下午4:33

def data_resample():
    """
    将/home/aone/lisen/code/FastSpeech2/preprocessed_data/AISHELL3Tiny中的train.txt和val.txt挑选少部分内容用于训练
    :return:
    """
    src_file_path = "/home/aone/lisen/code/FastSpeech2/preprocessed_data/AISHELL3Tiny/val.txt"
    tar_file_path = "/preprocessed_data/AISHELL3Tiny/val.txt"
    tar_file = open(tar_file_path, "w", encoding='utf8')
    with open(src_file_path) as src_file:
        for line in src_file:
            if line[:7] == 'SSB0005' or line[:7] == 'SSB0009':
                tar_file.write(line)



if __name__ == '__main__':
    data_resample()
