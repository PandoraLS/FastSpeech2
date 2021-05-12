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

if __name__ == "__main__":
    # speaker_id: 0-217
    # man: 212 211 197 192 187 184 176 172 162 160 151
    # female: 60 126
    # model_call(text = "你们", speaker_id = 145, restore_step = 900000, mode = "single", pro_path = "config/AISHELL3/preprocess.yaml", model_path = "config/AISHELL3/model.yaml", train_path = "config/AISHELL3/train.yaml")
    # 使用pycharm内部就无法使用命令行了
    model_call(text = "今晚八点比赛直播我们不见不散", speaker_id = 212, restore_step = 900000, mode = "single", pro_path = "config/AISHELL3/preprocess.yaml", model_path = "config/AISHELL3/model.yaml", train_path = "config/AISHELL3/train.yaml")


