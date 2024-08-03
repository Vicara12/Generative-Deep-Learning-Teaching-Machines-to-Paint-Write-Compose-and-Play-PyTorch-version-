from time import time


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