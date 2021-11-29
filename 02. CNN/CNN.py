import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

# Converting classes into numeric format
df['numeric_class'] = df['Class'].astype('category').cat.codes
df


def train_val_split(df):
    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)

    train_df = df[:int(df['ID'].count() * 0.8)]
    val_df = df[4348:]

    return train_df, val_df

train_df, val_df = train_val_split(df)
train_df.shape, val_df.shape

import cv2
import sys


def drawProgressBar(current, total, string='', barLen=20):
    percent = current / total
    arrow = ">"
    if percent == 1:
        arrow = ""

    sys.stdout.write("\r")
    sys.stdout.write("Progress: [{:<{}}] {}/{}".format("=" * int(barLen * percent) + arrow,
                                                       barLen, current, total) + string)
    sys.stdout.flush()


def get_audio_same_len(wav, sr):
    if wav.shape[0] < 4 * sr:
        wav = np.pad(wav, int(np.ceil((4 * sr - wav.shape[0]) / 2)), mode='reflect')
    wav = wav[:4 * sr]

    return wav


def get_melspectrogram_db(wav, sr):
    wav = get_audio_same_len(wav, sr)

    spec = librosa.feature.melspectrogram(wav, sr, n_fft=2048, hop_length=512,
                                          n_mels=128, fmin=20, fmax=8300)

    spec = librosa.power_to_db(spec, top_db=80)
    return spec

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def standard_norm(spec):
    mMscaler = MinMaxScaler()
    sdscaler = StandardScaler()

    spec = sdscaler.fit_transform(spec)
    spec = mMscaler.fit_transform(spec)
    spec_scaled = spec*255

    return spec_scaled


