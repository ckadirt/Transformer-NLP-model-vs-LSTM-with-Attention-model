from utils.process_text import create_tokenizer, text_to_token, text_to_token_plus, token_to_text
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
class LSTM_Dataset(Dataset):
  def __init__(self, csvpath):
    self.csvpath = csvpath
    self.data = pd.read_csv(csvpath)
    self.tokenizer = create_tokenizer()

  def __len__(self):
    return len(self.data)

  def convert_to_tokens(self, dataframe_row):
    tokens = text_to_token_plus(dataframe_row['dialog'],self.tokenizer)
    return tokens
  def __getitem__(self, index):
    row = self.data.iloc[index]
    tokens = self.convert_to_tokens(row)
    #print(tokens['input_ids'])
    data = [torch.tensor(x).float() for x in tokens['input_ids']]
    try:
      separator = random.randint(1,len(data)-1)
      t = data[0:separator]
      x = torch.tensor([])
      for sentence in t:
        x = torch.cat([x,sentence], dim = -1)
      y = data[separator]
    except:
      print(data)
      x = data[-1]
      y = torch.tensor([0])
    #x = x.unsqueeze(dim = 1)
    
    pad_x = torch.zeros((1024), device=x.device, dtype=x.dtype)
    pad_x[:x.shape[0]] = x
    #print(pad_x.shape)
    pad_y = torch.zeros((128), device=y.device, dtype=y.dtype)
    pad_y[:y.shape[0]] = y
    return {'x' : pad_x, 'y' : pad_y}

