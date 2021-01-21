import torch
from torchtext.vocab import build_vocab_from_iterator
import os.path
import time
import torch.nn as nn
import numpy as np
import math

from values import MODEL_SAVE_NAME
from dataset import setup_dataset
from model import TransformerModel, PositionalEncoding
from waveform import setup_raw_audio, data_process, batchify
from train import train, evaluate

DOWNLOAD_DATASET = False
EXTRACT_DATASET = False
SKIP_SETUP_DATA = False

setup_dataset(DOWNLOAD_DATASET, EXTRACT_DATASET)

train_data = []
val_data = []
data = []

if not SKIP_SETUP_DATA:
    res_data = setup_raw_audio()
    train_data = res_data['train_data']
    val_data = res_data['val_data']
    data = res_data['data']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = build_vocab_from_iterator(iter(data))

train_data = data_process(vocab, iter(train_data))
val_data = data_process(vocab, iter(val_data))

# Setup parameters
bptt = 35
batch_size = 32
eval_batch_size = 16
train_data = batchify(train_data, batch_size, device)
val_data = batchify(val_data, eval_batch_size, device)
ntokens = len(vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 32  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
lr = 0.1  # learning rate
epochs = 15  # The number of epochs

# Setup Model
model = TransformerModel(ntokens, emsize, nhead, nhid,
                         nlayers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)

best_val_loss = float("inf")
best_model = None
epoch_losses = np.array([])
epoch = 1

if os.path.isfile(MODEL_SAVE_NAME):
    print('loading checkpoint...')
    checkpoint = torch.load(MODEL_SAVE_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    saved_loss = checkpoint['loss']

    if saved_loss < best_val_loss:
        best_val_loss = saved_loss

while epoch < epochs + 1:
    epoch_start_time = time.time()
    train(model, bptt, device, train_data, optimizer, criterion, ntokens, scheduler, epoch)
    val_loss = evaluate(model, model, val_data, bptt, device, ntokens, criterion)
    epoch_losses = np.append(epoch_losses, val_loss)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, MODEL_SAVE_NAME)

    epoch = epoch + 1
    scheduler.step()