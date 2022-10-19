from process_text import create_tokenizer, text_to_token, text_to_token_plus, token_to_text
import pandas as pd
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

class LSTM_single(nn.Module):
  def __init__(self, out_size, embedding_size = 300):
    super(LSTM_single, self).__init__()
    self.out_size = out_size
    self.params = self.create_a_layer(embedding_size, out_size)

  def create_a_layer(self, input_size, output_size):
    input_size = input_size + output_size
    w_c = nn.Parameter(torch.empty(output_size, input_size), requires_grad = True)
    w_u = nn.Parameter(torch.empty(output_size, input_size), requires_grad = True)
    w_f = nn.Parameter(torch.empty(output_size, input_size), requires_grad = True)
    w_o = nn.Parameter(torch.empty(output_size, input_size), requires_grad = True)

    b_c = nn.Parameter(torch.empty(1, output_size), requires_grad = True)
    b_u = nn.Parameter(torch.empty(1, output_size), requires_grad = True)
    b_f = nn.Parameter(torch.empty(1, output_size), requires_grad = True)
    b_o = nn.Parameter(torch.empty(1, output_size), requires_grad = True)

    nn.init.xavier_uniform_(w_c)
    nn.init.xavier_uniform_(w_u)
    nn.init.xavier_uniform_(w_f)
    nn.init.xavier_uniform_(w_o)
    nn.init.xavier_uniform_(b_c)
    nn.init.xavier_uniform_(b_u)
    nn.init.xavier_uniform_(b_f)
    nn.init.xavier_uniform_(b_o)

    return {'w_c': w_c,'w_u': w_u, 'w_f': w_f, 'w_o': w_o,'b_c': b_c,'b_u': b_u,'b_f':b_f, 'b_o': b_o}

  def forward(self, x_i, a_0, c_0):
    #pass the x embedded (batch, words, embedded)
    params = sorted(self.params.items())
    b_c, b_f, b_o, b_u, w_c, w_f, w_o, w_u = [v[1] for v in params]
    a_f = torch.zeros(x_i.shape[0],x_i.shape[1],self.out_size)
    for tens in range(0,x_i.shape[1]):
      x = x_i[:,tens,:]
      x_a_concat = torch.cat([a_0, x], dim = 1).transpose(1,0)
      candidate = torch.tanh(torch.matmul(w_c, x_a_concat).transpose(1,0) + b_c)
      g_u = torch.sigmoid(torch.matmul(w_u, x_a_concat).transpose(1,0)+ b_u)
      g_f = torch.sigmoid(torch.matmul(w_f, x_a_concat).transpose(1,0)+ b_f)
      g_o = torch.sigmoid(torch.matmul(w_o, x_a_concat).transpose(1,0)+ b_o)
      c_1 = torch.mul(g_u, candidate) + torch.mul(g_f,c_0)
      a_1 = torch.mul(g_o, torch.tanh(c_1))
      
      a_f[:,tens,:] = a_1

      a_0, c_0 = a_1, c_1

    #print(a_f.shape)
    return a_f, a_0, c_0
    
class LSTM_singlebi(nn.Module):
  def __init__(self, out_size, embedding_size = 300):
    super(LSTM_singlebi, self).__init__()
    self.out_size = out_size
    self.LSTM_model = LSTM_single(out_size, embedding_size = embedding_size)

  def forward(self, x, a0, c0):
    # pass the x value embedded (batch_size, words, embedding_size)
    aright, a_0r, c_0r = self.LSTM_model(x, a0, c0)
    aleft, a_0l, c_0l = self.LSTM_model(torch.flip(x, (1,)),a_0r, c_0r)
    return torch.cat([aright, aleft], dim = 2), (a_0r,c_0r, a_0l,c_0l)

class Depth_LSTM(nn.Module):
  def __init__(self, out_size, depth = 2, embedding_size = 300, vocab_size = 30522):
    super(Depth_LSTM, self).__init__()
    self.out_size = out_size
    self.depth = depth
    self.embedding_size = embedding_size
    self.layers = []
    self.linears = []
    self.relu = nn.LeakyReLU()
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    for a in range(depth):
      if(a == 0):
        self.layers.append(LSTM_singlebi(self.out_size, embedding_size=embedding_size))
      else:
        self.layers.append(LSTM_singlebi(self.out_size, embedding_size=out_size))
    for a in range(depth):
      self.linears.append(nn.Linear(self.out_size * 2, self.out_size))
  
  def forward(self, x, a, c):
    x = self.embedding(x)
    for index, layer in enumerate(self.layers):
      print(x.shape, 'first')
      x, rest = layer(x, a, c)
      print(x.shape, 'second')
      x = self.relu(self.linears[index](x))
      print(x.shape, 'third')

    return x