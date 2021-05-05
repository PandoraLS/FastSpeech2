# -*- coding: utf-8 -*-
# @Time    : 2021/5/5 下午1:14

"""
数融的数据修改为可以训练的格式，并且将汉字修改为拼音
"""
import string
import re, os, shutil
from pypinyin import lazy_pinyin, Style
from zhon.hanzi import punctuation

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

def remove_punctuation(sentence):
    """
    去除sentence中的所有中英文标点
    :param sentence (str):
    :return:
    """
    # 去除中文标点
    res = re.sub(r"[%s]+" % punctuation, "", sentence)

    # 去除英文标点
    exclude = set(string.punctuation)
    out = ''.join(ch for ch in res if ch not in exclude)
    return out

def phonetic(sentence):
    """
    对一句汉字注音
    :param sentence: 字符串
    :return:
    """
    # 不考虑多音字() 5音表示中性音
    res = lazy_pinyin(sentence, style=Style.TONE3, neutral_tone_with_five=True)
    return res

def prep_shurong_raw_data():
    """
    处理一个人的录音, 将汉字变为拼音, 将该speaker的文件夹结构改为AISHELL3的格式
    :param speaker_dir: 原始speaker的文件夹
    :param target_root_dir: 目标文件夹根目录
    :return:
    """
    wav_dir = "/home/aone/lisen/dataset/shurong/speech_data_target"
    wav_list = os.listdir(wav_dir)
    txt_file_path = "/home/aone/lisen/code/FastSpeech2/data_utils/shurong_label_tar.txt"
    target_speaker_dir = "/home/aone/lisen/dataset/shurong/shurong_target"
    creat_dir(target_speaker_dir)


    with open(txt_file_path) as txt_file:
        for line in txt_file:
            num_str, sentence = line.strip().split('\t')
            pinyin_of_sentence = ' '.join(phonetic(sentence))
            lab_file_name = 'shurong' + num_str + '.lab'
            for wav in wav_list:
                if 'shurong'+num_str == wav[:-4]:
                    forceCopyFile(os.path.join(wav_dir, wav), os.path.join(target_speaker_dir, wav))
                    tar_lab_file = open(os.path.join(target_speaker_dir, lab_file_name), 'w', encoding='utf8')
                    tar_lab_file.write(pinyin_of_sentence)
                    tar_lab_file.write('\n')
                    tar_lab_file.close()


def prep_dataset():
    """
    将甲方提供的数融数据整理为librispeech格式的
    :return:
    """
    wav_dir = "/home/aone/lisen/dataset/shurong/speech_data"
    wav_list = os.listdir(wav_dir)
    wav_list.sort()
    cnt = 0
    tar_dir = "/home/aone/lisen/dataset/shurong/speech_data_target/"
    creat_dir(tar_dir)
    for wav in wav_list:
        cnt += 1
        wav_path = os.path.join(wav_dir, wav)
        wav_new_name = 'shurong' + str(cnt).zfill(5) + '.wav'
        tar_wav_path = tar_dir + wav_new_name
        forceCopyFile(wav_path, tar_wav_path)

    print()



def remove_file_punctuation():
    txt_file = "/home/aone/lisen/code/FastSpeech2/data_utils/shurong_label.txt"
    tar_txt = "/home/aone/lisen/code/FastSpeech2/data_utils/shurong_label_tar.txt"
    tar_file = open(tar_txt, 'w', encoding='utf8')
    src_lines = open(txt_file).readlines()
    cnt = 0
    for line in src_lines:
        cnt += 1
        line = line.strip()
        new_line = remove_punctuation(line)
        tar_file.write(str(cnt).zfill(5) + '\t' + new_line)
        tar_file.write('\n')


if __name__ == '__main__':
    pass
    # remove_file_punctuation()
    prep_shurong_raw_data()
