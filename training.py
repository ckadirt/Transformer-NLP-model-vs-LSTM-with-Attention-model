test_datas = LSTM_Dataset('data_train.csv')
dataloadtt = DataLoader(test_datas, batch_size = 1)
for batch_test in dataloadtt:
  print(batch_test['y'])
  print(token_to_text(batch_test['y'].int(), test_datas.tokenizer))
  break

device = 'cuda'
encoder = Depth_LSTM(400).to(device)


criterion = nn.BCELoss()
optimizer_enc = torch.optim.Adam(encoder.parameters())


decoder = Attention_Decoder(128,10000,400).to(device)

optimizer_dec = torch.optim.Adam(decoder.parameters())



def batch_train(encoder, decoder, batch, criterion, optimizer_enc, optimizer_dec, device):
  
  x = batch['x'].to(device).int()
  a_init = torch.zeros(x.shape[0],encoder.out_size).to(device)
  c_init = torch.zeros(x.shape[0],encoder.out_size).to(device)
  y = batch['y'].to(device)

  optimizer_enc.zero_grad()
  optimizer_dec.zero_grad()

  y_mid = encoder(x, a_init, c_init)
  s_prev = torch.zeros(x.shape[0],decoder.output_features)
  y_hat = decoder(y_mid, s_prev, y.shape[-1])
  print('Printing y_hat ****************', y_hat, y.shape[-1])
  y = torch.nn.functional.one_hot(y, num_classes = 10000)
  loss = criterion(y, y_hat)
  loss.backward()
  optimizer_enc.step()
  optimizer_dec.step()


batch_train(encoder, decoder, batch_test, criterion, optimizer_enc, optimizer_dec, device)