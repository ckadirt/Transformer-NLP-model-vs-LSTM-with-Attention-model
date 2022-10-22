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
      #print(x.shape, 'first')
      x, rest = layer(x, a, c)
      #print(x.shape, 'second')
      x = self.relu(self.linears[index](x))
      #print(x.shape, 'third')

    return x

class Context_generator(nn.Module):
  def __init__(self, s_feat, a_feat, word_len):
    super(Context_generator, self).__init__()
    self.middle = 1000
    self.linear1 = nn.Linear(s_feat+a_feat, self.middle)
    self.linear2 = nn.Linear(self.middle, 1)
    
    self.relu = nn.ReLU()
    self.tahn = nn.Tanh()
    self.softmax = nn.Softmax(dim = 1)
  def forward(self, s_prev, a):
    # a shape = [batch, num_words, features]
    # s shape = [batch, features]
    s_prev = s_prev.view(s_prev.shape[0],1, s_prev.shape[1]).repeat(1,a.shape[1],1)
    concated = torch.cat([a, s_prev], dim = -1)
    alphas = self.tahn(self.linear1(concated))
    alphas = self.relu(self.linear2(alphas))
    alphas = self.softmax(alphas)
    context = torch.bmm(alphas.transpose(1,2), a)
    return context, alphas


class Attention_Decoder(nn.Module):
  def __init__(self, word_len, output_features, a_features):
    super(Attention_Decoder, self).__init__()
    self.word_len = word_len
    self.output_features = output_features
    self.a_features = a_features
    self.get_context = Context_generator(s_feat = output_features, a_feat = a_features, word_len = word_len)
    self.lstm = LSTM_single(output_features, embedding_size = a_features)

  def forward(self, a, s_prev, word_output):
    # a = [batches, num_words, output_size features]
    #s_prev = [batches, output_features]
    #word_output = int, num of the word outputs
    final_as = torch.zeros(a.shape[0],word_output,self.output_features)
    s_prev = s_prev
    c0 = torch.zeros(a.shape[0],s_prev.shape[-1])
    for word in range(word_output):
      context, alphas = self.get_context(s_prev, a)
      #print(context.shape, s_prev.shape, c0.shape)
      s_prev, s_prevlast, c0 = self.lstm(context, s_prev, c0)
      s_prev = s_prev.view(a.shape[0],s_prev.shape[-1])
      #print(s_prev.shape, final_as.shape)
      final_as[:,word,:] = s_prev
