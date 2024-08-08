from time import time
import torch


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
  t_ini = time()
  for epoch in range(epochs):
    training_loss = 0.0
    valid_loss = 0.0
    model.train()
    i = 0
    for batch in train_loader:
      i += 1
      optimizer.zero_grad()
      inputs, _ = batch
      inputs = inputs.to(device)
      output = model(inputs)
      loss = loss_fn(output, inputs)
      loss.backward()
      optimizer.step()
      training_loss += loss.data.item() * inputs.size(0)
    training_loss /= len(train_loader.dataset)
    model.eval()
    for batch in val_loader:
      inputs, _ = batch
      inputs = inputs.to(device)
      output = model(inputs)
      loss = loss_fn(output, inputs)
      valid_loss += loss.data.item() * inputs.size(0)
    valid_loss /= len(val_loader.dataset)
    print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}'.format(epoch, training_loss, valid_loss))
  print(f"Trainning took {time() - t_ini} seconds")

def trainVariational(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
  beta = 1/100000 # mixing of kn and other loss
  t_ini = time()
  kl_loss_fn = lambda mean, log_var: -0.5*torch.sum(torch.ones(mean.shape) + log_var - mean**2 - torch.exp(log_var))
  for epoch in range(epochs):
    training_loss = 0.0
    valid_loss = 0.0
    model.train()
    i = 0
    for batch in train_loader:
      i += 1
      optimizer.zero_grad()
      inputs, _ = batch
      inputs = inputs.to(device)
      z_mean, z_log_var, output = model(inputs, True)
      loss_custom = loss_fn(output, inputs)
      loss_knd = kl_loss_fn(z_mean, z_log_var)
      # loss = loss_custom
      loss = (1-beta)*loss_custom + beta*loss_knd
      loss.backward()
      optimizer.step()
      training_loss += loss.data.item() * inputs.size(0)
    training_loss /= len(train_loader.dataset)
    model.eval()
    for batch in val_loader:
      inputs, _ = batch
      inputs = inputs.to(device)
      output = model(inputs)
      loss = loss_fn(output, inputs)
      valid_loss += loss.data.item() * inputs.size(0)
    valid_loss /= len(val_loader.dataset)
    print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}'.format(epoch, training_loss, valid_loss))
  print(f"Trainning took {time() - t_ini} seconds")