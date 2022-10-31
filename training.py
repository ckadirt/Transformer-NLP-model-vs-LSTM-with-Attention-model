from utils.process_text import create_tokenizer, text_to_token, text_to_token_plus, token_to_text
import pandas as pd
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

from utils.create_dataset import LSTM_Dataset
from utils.LSTM_models import Depth_LSTM, Attention_Decoder
from utils.process_text import token_to_text

test_datas = LSTM_Dataset('data_train.csv')
dataloadtt = DataLoader(test_datas, batch_size = 2)
for batch_test in dataloadtt:
  print(batch_test['y'])
  print(token_to_text(batch_test['y'].int(), test_datas.tokenizer))
  break

device = 'cpu'
encoder = Depth_LSTM(400).to(device)


criterion = nn.BCELoss()
optimizer_enc = torch.optim.Adam(encoder.parameters())


decoder = Attention_Decoder(128,30522,400)
optimizer_dec = torch.optim.Adam(decoder.parameters())

def batch_train(encoder, decoder, batch, criterion, optimizer_enc, optimizer_dec, device):
  
  x = batch['x'].to(device).int()
  a_init = torch.zeros(x.shape[0],encoder.out_size).to(device)
  c_init = torch.zeros(x.shape[0],encoder.out_size).to(device)
  y = batch['y'].to(device)
  y = F.one_hot(y, num_classes=decoder.output_features)
  optimizer_enc.zero_grad()
  optimizer_dec.zero_grad()

  y_mid = encoder(x, a_init, c_init)
  s_prev = torch.zeros(x.shape[0],decoder.output_features)
  y_hat = decoder(y_mid, s_prev, y.shape[-1])
  print('Printing y_hat ****************', y_hat, y.shape[-1])
  loss = criterion(y, y_hat.reshape(y_hat.shape[0], y_hat.shape[1]))
  loss.backward()
  optimizer_enc.step()
  optimizer_dec.step()


batch_train(encoder, decoder, batch_test, criterion, optimizer_enc, optimizer_dec, device)