# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 10/18/2022

import re
import pickle
import unicodedata
from .text2mel import text2mel
from .mel2wav import mel2wave
from .config import FLAGS
import soundfile as sf


def nat_normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    sil = FLAGS.special_phonemes[FLAGS.sil_index]
    text = re.sub(r"[\n.,:]+", f" {sil} ", text)
    text = text.replace('"', " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,:;?!]+", f" {sil} ", text)
    text = re.sub("[ ]+", " ", text)
    text = re.sub(f"( {sil}+)+ ", f" {sil} ", text)
    return text.strip()


def tts(text, path,silence_duration=0.2, sample_rate=16000):
    text = nat_normalize_text(text)
    mel = text2mel(text, silence_duration)
    wave = mel2wave(mel)
    sf.write(path, wave, samplerate=sample_rate)




