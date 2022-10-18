from process_text import create_tokenizer, text_to_token, text_to_token_plus, token_to_text
import pandas as pd
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

class LSTM_uni(nn.Module):
  def __init__(self, out_size, depth, size_vocab = 30522, embedding_size = 300):
    super(LSTM_uni, self).__init__()
    self.out_size = out_size
    self.depth = depth
    self.layers = []
    self.embeddings = nn.Embedding(size_vocab, embedding_size)
    for a in range(0,depth):
      if (a == 0):
        self.layers.append(self.create_a_layer(embedding_size, self.out_size, first = True))
      else:
        self.layers.append(self.create_a_layer(embedding_size, self.out_size, first = False))

  def create_a_layer(self, input_size, output_size, first = True):
    if (first):
      input_size = output_size + input_size
    else:
      input_size = output_size * 2
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
    #x = shape(1, input_size)
    #a_0 = shape(layers, embedding_size)
    #c_0 = shape(layers, embedding_size)
    x = self.embeddings(x_i[0].int()) #(num_words, embeddings)
    for layer in range(0, self.depth):
      params = sorted(self.layers[layer].items())
      b_c, b_f, b_o, b_u, w_c, w_f, w_o, w_u = [v[1] for v in params]
      for word in range(0,x.size(0)):
        x_a_concat = torch.cat([x[word].unsqueeze(0), a_0[layer].unsqueeze(0)], dim = 1).transpose(1,0)
        candidate = torch.tanh(torch.matmul(w_c, x_a_concat).transpose(1,0) +b_c)
        g_u = torch.sigmoid(torch.matmul(w_u, x_a_concat).transpose(1,0)+ b_u)
        g_f = torch.sigmoid(torch.matmul(w_f, x_a_concat).transpose(1,0)+ b_f)
        g_o = torch.sigmoid(torch.matmul(w_o, x_a_concat).transpose(1,0)+ b_o)

        #print(candidate.shape, g_u.shape, g_f.shape, c_0[index].unsqueeze(0).transpose(1,0).shape)
        c_1 = torch.mul(g_u, candidate) + torch.mul(g_f,c_0[layer].unsqueeze(0))
        a_1 = torch.mul(g_o, torch.tanh(c_1))
        a_0[layer] = a_1
        c_0[layer] = c_1
        x[word] = a_1
        #print(candidate.shape, g_u.shape, g_f.shape, c_1.shape, a_1.shape)
    return x, a_0, c_0

class LSTM_singlebi(nn.Module):
  def __init__(self, out_size, embedding_size = 300):
    super(LSTM_singlebi, self).__init__()
    self.out_size = out_size
    self.LSTM_model = LSTM_single(int(out_size/2), embedding_size = embedding_size)

  def forward(self, x, a0, c0):
    # pass the x value embedded (batch_size, words, embedding_size)
    aright, a_0r, c_0r = self.LSTM_model(x, a0, c0)
    aleft, a_0l, c_0l = self.LSTM_model(torch.flip(x, (1,)),a_0r, c_0r)
    return torch.cat([aright, aleft], dim = 2), (a_0r,c_0r, a_0l,c_0l)