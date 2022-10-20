from process_text import create_tokenizer, text_to_token, text_to_token_plus, token_to_text
import pandas as pd
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

class Attention_model(nn.Module):
  def __init__(self):
    super(Attention_model, self).__init__()

  def forward(self, x):
    return x