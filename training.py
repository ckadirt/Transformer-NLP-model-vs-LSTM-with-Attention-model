from utils.process_text import create_tokenizer, text_to_token, text_to_token_plus, token_to_text
import pandas as pd
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

data_train = pd.read_csv('daily-train.csv')
data_validation = pd.read_csv('daily-validation.csv')
data_test = pd.read_csv('daily-test.csv')
data_train['dialog']
def get_row_len(row):
    return len(row['dialog'])
data_train['len_dialog'] = data_train.apply(get_row_len, axis = 1)
data_train = data_train[data_train['len_dialog']>1020]

dd = LSTM_Dataset('daily-train.csv')
dl = DataLoader(dd, batch_size = 16, shuffle = False)
for example in dl:
  #print(example['x'].shape)
  print(example['x'].shape, example['y'].shape)

  break