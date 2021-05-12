# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 下午6:06

import re
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

def model_call(text, speaker_id: int, restore_step: int, mode, pro_path, model_path, train_path, pitch_control = 1.0, energy_control = 1.0, duration_control = 1.0):
    # Check source texts
    source = None
    if mode == "batch":
        assert source is not None and text is None
    if mode == "single":
        assert source is None and text is not None

    # Read config
    preprocess_config = yaml.load(open(pro_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_path, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(restore_step, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if mode == "batch":
        # Get dataset
        dataset = TextDataset(source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if mode == "single":
        ids = raw_texts = [text[:100]]
        speakers = np.array([speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = pitch_control, energy_control, duration_control

    synthesize(model, restore_step, configs, vocoder, batchs, control_values)

def model_init(restore_step: int,pro_path, model_path, train_path, pitch_control = 1.0, energy_control = 1.0, duration_control = 1.0):
    # Read config
    preprocess_config = yaml.load(open(pro_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_path, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(restore_step, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    control_values = pitch_control, energy_control, duration_control
    return model, restore_step, configs, vocoder, control_values

# ------------------------------------------------------------------------------------------------
import os
import shutil
def forceCopyFile(sfile, dfile):
    """
    将文件覆盖拷贝
    Args:
        sfile: 源文件path
        dfile: 目标文件path
    Returns:
    """
    if os.path.isfile(sfile):
        shutil.copy2(sfile, dfile)

def synthesize_speaker_id(speaker_id):
    """
    生成对应id的话音
    :return:
    """
    txt_list = [
        "爱丽丝开始觉得有点无聊了",
        "姐姐在看书而爱丽丝无事可做",
        "里面既没有图画",
        "她不时看看姐姐的书",
        "她和姐姐正坐在树下",
        "也没有对话"
    ]
    result_root = "/home/aone/lisen/code/FastSpeech2/output/result/AISHELL3"
    synthe_root = os.path.join(result_root, str(speaker_id))
    if not os.path.exists(synthe_root):
        os.makedirs(synthe_root)

    for txt_item in txt_list:
        model_call(text=txt_item, speaker_id=speaker_id, restore_step=900000, mode="single",
                   pro_path="config/AISHELL3/preprocess.yaml", model_path="config/AISHELL3/model.yaml",
                   train_path="config/AISHELL3/train.yaml")

    files = os.listdir(result_root)
    for file in files:
        if file[-4:] == '.png' or file[-4:] == '.wav':
            forceCopyFile(os.path.join(result_root, file), os.path.join(synthe_root, file))
            os.remove(os.path.join(result_root, file))

if __name__ == "__main__":
    pass
    # speaker_id: 0-217
    # man: 212 211 197 192 187 184 176 172 162 160 151
    # female: 60 126
    # model_call(text = "你们", speaker_id = 145, restore_step = 900000, mode = "single", pro_path = "config/AISHELL3/preprocess.yaml", model_path = "config/AISHELL3/model.yaml", train_path = "config/AISHELL3/train.yaml")
    # 使用pycharm内部就无法使用命令行了
    # txt_list = [
    #     "爱丽丝开始觉得有点无聊了",
    #     "姐姐在看书而爱丽丝无事可做",
    #     "里面既没有图画",
    #     "她不时看看姐姐的书",
    #     "她和姐姐正坐在树下",
    #     "也没有对话"
    # ]
    # model_call(text = "今晚八点比赛直播，我们不见不散", speaker_id = 212, restore_step = 900000, mode = "single", pro_path = "config/AISHELL3/preprocess.yaml", model_path = "config/AISHELL3/model.yaml", train_path = "config/AISHELL3/train.yaml")

    # synthesize_speaker_id(60)
    # synthesize_speaker_id(126)
    # synthesize_speaker_id(144)
    # synthesize_speaker_id(22)
    # synthesize_speaker_id(154)
    # synthesize_speaker_id(107)
    # synthesize_speaker_id(216)

    # 长段话转录
    long_sentence = "爱丽丝开始觉得有点无聊了 她和姐姐正坐在树下 姐姐在看书 而爱丽丝无事可做 她不时看看姐姐的书 里面既没有图画 也没有对话 一本书没有图画和对话有什么用呢 爱丽丝想 她想找点什么事儿做做 可天气很热 她觉得又因又无聊 正坐在那儿想事 忽然 一只长着粉红眼睛的白兔跑过她身边 看到一只兔子真没有什么可奇怪的 兔子说话时爱丽丝居然也不觉得太奇怪 兔子说 噢 天哪 噢 天哪 我要迟到了 后来爱丽丝想起这事觉得有点儿奇怪 但当时她并不觉得有什么奇怪然后兔子从自己的口袋里掏出一块表 看了看 赶紧走了 爱丽丝立刻跳了起来 我从未见过有口袋的兔子 或者兔子掏出一块手表来 她想 她跟在兔子后面很快跑过田野 她也没停下来想一想 当兔子跑进一个大的兔子洞时 爱丽丝立即跟了进去 走了一小段 兔子洞突然向下转 直深入地下 爱丽丝不由自主地掉了下去"
    model_call(text=long_sentence, speaker_id=184, restore_step=900000, mode="single",
               pro_path="config/AISHELL3/preprocess.yaml", model_path="config/AISHELL3/model.yaml",
               train_path="config/AISHELL3/train.yaml")


