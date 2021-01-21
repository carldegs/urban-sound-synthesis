import csv
from values import DATASET_FOLDER
from tqdm import tqdm_notebook as tqdm
import librosa.display
import librosa
import numpy as np
import pandas as pd
import torch
import io
CUDA_LAUNCH_BLOCKING = "1"

SOF = 299.
EOF = 99.

def get_waveform(idx, rows):
    row = rows[idx]
    filename = row[0]
    fold = row[1]
    wav, sr = librosa.load(
        f'{DATASET_FOLDER}/audio/fold{fold}/{filename}', sr=8000)
    wav = np.around(wav, 4)
    wav = np.append(wav, EOF)
    wav = np.insert(wav, 0, SOF)

    return wav


def get_audio_data_from_csv(category='dog_bark'):
    df = pd.read_csv(F'{DATASET_FOLDER}/metadata/UrbanSound8K.csv')
    groupedData = df[['slice_file_name', 'fold', 'class']
                     ].groupby('class').apply(np.array)
    rows = groupedData[category]

    audio_data = []

    for idx in tqdm(range(int(len(rows)))):
        audio_wav_sr_data = get_waveform(idx, rows)

        audio_data.append(audio_wav_sr_data)

    return audio_data


def setup_raw_audio():
    data = get_audio_data_from_csv()
    b = int(len(data)*0.8)
    train_data = data[0:b]
    val_data = data[b+1:len(data)]

    res = dict()
    res['train_data'] = train_data
    res['val_data'] = val_data
    res['data'] = data

    return res

def waveform_detokenizer(iter):
    arr = [np.round((np.array(item) / 10000) - 1, 4) for item in iter]
    return arr

def data_process(vocab, iter):
    data = [torch.tensor([vocab[token] for token in row], dtype=torch.long) for row in iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data, bsz, device):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()

    return data.to(device)

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target