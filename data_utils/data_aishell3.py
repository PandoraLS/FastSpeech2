# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 下午1:43


"""
将aishell3由原始格式处理为适配FastSpeech2使用的格式, 只处理train/中的人员
"""
import os, shutil

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

def isAlNum(word):
    """
    系统自带的isalnum()会把中文汉字也判为True
    :param word:
    :return:
    """
    try:
        return word.encode('ascii').isalnum()
    except UnicodeEncodeError:
        return False

def prep_aishell3train():
    """
    处理aishell3train部分
    :return:
    """
    train_wav_dir = "/home/aone/lisen/dataset/data_aishell3/train/wav"
    txt_path = "/home/aone/lisen/dataset/data_aishell3/train/content.txt"
    target_dir = "/home/aone/lisen/dataset/AISHELL3Train"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    speakers = os.listdir(train_wav_dir)

    with open(txt_path) as txt_file:
        for line in txt_file:
            wav_name, transcription = line.strip().split('\t')
            transcription_pinyin = ' '.join([item for item in transcription.split() if isAlNum(item)])
            speaker_name = wav_name[:7]
            if speaker_name in speakers:
                creat_dir(os.path.join(target_dir, speaker_name)) # 创建目标文件夹路径
                src_wav_file = os.path.join(train_wav_dir, speaker_name)  + '/' + wav_name
                target_wav_file = os.path.join(target_dir, speaker_name) + '/' + wav_name
                forceCopyFile(src_wav_file, target_wav_file)
                tar_lab_file = open(target_wav_file[:-4] + '.lab', 'w', encoding='utf8')
                tar_lab_file.write(transcription_pinyin + '\n')
                tar_lab_file.close()

if __name__ == '__main__':
    pass
    prep_aishell3train()