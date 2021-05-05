# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 下午8:12

"""
用于处理aone_tts_dataset/的数据, 将汉字变成注音的格式，同时文件结构修改为AISHELL3的格式
pypinyin:https://pypi.org/project/pypinyin/
"""
import os, shutil
from pypinyin import lazy_pinyin, Style

def creat_dir(d):
    """
    新建文件夹
    Args:
        d: 文件夹路径
    Returns:
    """
    if not os.path.exists(d):
        os.makedirs(d)

def forceCopyFile(sfile, dfile):
    """
    将文件覆盖拷贝
    Args:
        sfile: 源文件path 比如xxxx/xxxx/xxxx.txt
        dfile: 目标文件path 比如 yyyy/yyyy/yyyy.txt
    Returns:
    """
    if os.path.isfile(sfile):
        shutil.copy2(sfile, dfile)

def phonetic(sentence):
    """
    对一句汉字注音
    :param sentence: 字符串
    :return:
    """
    # 不考虑多音字() 5音表示中性音
    res = lazy_pinyin(sentence, style=Style.TONE3, neutral_tone_with_five=True)
    return res

def prep_one_speaker(speaker_dir, target_root_dir):
    """
    处理一个人的录音, 将汉字变为拼音, 将该speaker的文件夹结构改为AISHELL3的格式
    :param speaker_dir: 原始speaker的文件夹
    :param target_root_dir: 目标文件夹根目录
    :return:
    """
    # speaker_dir = "/home/aone/lisen/dataset/aone_tts_dataset/chencong"
    root_dir, speaker_name = os.path.split(speaker_dir)
    txt_file_path = os.path.join(speaker_dir, speaker_name + '.txt')
    wav_dir = os.path.join(speaker_dir, speaker_name)
    wav_list = os.listdir(wav_dir)
    target_speaker_dir = os.path.join(target_root_dir, speaker_name)
    creat_dir(target_speaker_dir)

    with open(txt_file_path) as txt_file:
        for line in txt_file:
            num, sentence = line.strip().split('.')
            pinyin_of_sentence = ' '.join(phonetic(sentence))
            num_str = str(num).zfill(5) # 填充为5位
            lab_file_name = speaker_name + num_str + '.lab'
            for wav in wav_list:
                if num == wav[:-4]:
                    forceCopyFile(os.path.join(wav_dir, wav), os.path.join(target_speaker_dir, speaker_name + num_str + wav[-4:]))
                    tar_lab_file = open(os.path.join(target_speaker_dir, lab_file_name), 'w', encoding='utf8')
                    tar_lab_file.write(pinyin_of_sentence)
                    tar_lab_file.write('\n')
                    tar_lab_file.close()

def prep_speakers():
    """
    处理所有speaker的数据
    :return:
    """
    target_root_dir = '/home/aone/lisen/dataset/aone_tts_dataset_target'
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)

    src_root_dir = '/home/aone/lisen/dataset/aone_tts_dataset'
    speaker_dir_list = os.listdir(src_root_dir)
    for speaker_d in speaker_dir_list:
        src_speaker_dir = os.path.join(src_root_dir, speaker_d)
        prep_one_speaker(src_speaker_dir, target_root_dir)

def dataset_clean():
    """
    clean aone_tts_dataset_target dataset
    :return:
    """
    lab_wav_files_dir = "/home/aone/lisen/dataset/AISHELL3Train_HangTian/"

    clean_content = "/home/aone/lisen/dataset/AISHELL3Train_HangTian_TextGrid/unaligned.txt"
    with open(clean_content) as file:
        for line in file:
            file_name, _ = line.strip().split('\t')
            speaker_name = file_name[:-5]
            lab_file_to_remove_path = lab_wav_files_dir + speaker_name + '/' + file_name + '.lab'
            wav_file_to_remove_path = lab_wav_files_dir + speaker_name + '/' + file_name + '.wav'
            if os.path.exists(lab_file_to_remove_path):
                os.remove(lab_file_to_remove_path)
            if os.path.exists(wav_file_to_remove_path):
                os.remove(wav_file_to_remove_path)
            # print(line.strip())


if __name__ == '__main__':
    pass
    # print(phonetic('啦啦'))
    # prep_speakers()
    dataset_clean()
