CUDA_LAUNCH_BLOCKING="1"

import io
import torch
import pandas as pd
# from torchtext.utils import download_from_url, extract_archive
# from torchtext.data.utils import get_tokenizer
import numpy as np
from tqdm import tqdm_notebook as tqdm
import librosa
import librosa.display
from torchtext.vocab import build_vocab_from_iterator

SOF = 299.
EOF = 99.

DATASET_FOLDER = '/content/Audio_Classification_using_LSTM/UrbanSound8K'

def get_waveform(idx, rows):
    row = rows[idx]
    filename = row[0]
    fold = row[1]
    wav, sr = librosa.load(f'{DATASET_FOLDER}/audio/fold{fold}/{filename}', sr=8000)
    wav = np.around(wav, 4)
    # wav = librosa.util.fix_length(wav, SEQUENCE_LENGTH)
    # Add EOF
    wav = np.append(wav, EOF)
    wav = np.insert(wav, 0, SOF)

    return wav

def get_audio_data_from_csv(category='dog_bark'):
    df = pd.read_csv(f'{DATASET_FOLDER}/metadata/UrbanSound8K.csv')
    groupedData = df[['slice_file_name', 'fold', 'class']].groupby('class').apply(np.array)
    rows = groupedData[category]

    audio_data = []
    unique_wav_data = []

    for idx in tqdm(range(int(len(rows)))):
        audio_wav_sr_data = get_waveform(idx, rows)

        audio_data.append(audio_wav_sr_data)

    return audio_data

def waveform_detokenizer(iter):
    arr = [np.round((np.array(item) / 10000) - 1, 4) for item in iter]
    return arr