from process_text import create_tokenizer, text_to_token, text_to_token_plus, token_to_text
import pandas as pd
import torch
import torch.functional as F
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
    print(tokens['input_ids'])
    data = [torch.tensor(x).float() for x in tokens['input_ids']]
    separator = random.randint(1,len(data)-1)
    x = data[0:separator]
    y = data[separator]
    return {'x' : x, 'y' : y}