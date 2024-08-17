from time import time

import torch


def _trainGenerator(generator, discriminator, optimizer, loss_fn, iterations, batch_size, device):
  total_loss = 0
  generator.requires_gradient = True
  discriminator.requires_gradient = False
  generator.train()
  discriminator.eval()
  for _ in range(iterations):
    optimizer.zero_grad()
    inputs = torch.rand(size=(batch_size*2, generator.input_size)).to(device)
    gen_output = generator(inputs).to(device)
    disc = discriminator(gen_output)
    loss = loss_fn(disc, torch.ones(size=(batch_size*2,1)))
    loss.backward()
    optimizer.step()
    total_loss += loss
  generator.eval()
  inputs = torch.rand(size=(batch_size*2, generator.input_size)).to(device)
  gen_output = generator(inputs).to(device)
  disc = discriminator(gen_output)
  print(f"GENERATOR: achieved {round(100*sum(disc)/disc.shape[0],2)}% with loss = {total_loss/iterations}")

def _trainDiscriminator(generator, discriminator, optimizer, data_loader, loss_fn, iterations, batch_size, device):
  total_loss = 0
  generator.requires_gradient = False
  discriminator.requires_gradient = True
  generator.eval()
  discriminator.train()
  for batch in data_loader:
    real_samples, _ = batch
    optimizer.zero_grad()
    inputs = torch.rand(size=(batch_size, generator.input_size)).to(device)
    gen_output = generator(inputs)
    inputs = torch.concat([gen_output, real_samples], dim=0)
    labels = torch.concat([torch.zeros(size=(gen_output.shape[0],1)),
                           torch.ones(size=[real_samples.shape[0],1])], dim=0)
    indices = torch.randperm(labels.shape[0])
    inputs = inputs[indices].to(device)
    labels = labels[indices]
    outputs = discriminator(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss
  discriminator.eval()
  inputs = torch.rand(size=(batch_size, generator.input_size)).to(device)
  gen_output = generator(inputs).to(device)
  disc = discriminator(gen_output)
  n_incorrect = sum(disc)
  print(f"DISCRIMINATOR: achieved {round(100*(1-n_incorrect/disc.shape[0]),2)}% with loss = {total_loss/iterations}")


def train(generator, discriminator, optimizer, loss_fn, train_data_loader, ratio, loops, iterations, batch_size, device="cpu"):
  t_ini = time()
  loops_disc = max(1, round(1/ratio))
  loops_gen  = max(1, round(ratio))
  for i in range(loops):
    print(f" --- TRAINNING LOOP {i}/{loops} ---")
    for _ in range(loops_disc):
      _trainDiscriminator(generator, discriminator, optimizer, train_data_loader, loss_fn, iterations, batch_size, device)
    for _ in range(loops_gen):
      _trainGenerator(generator, discriminator, optimizer, loss_fn, iterations, batch_size, device)
  print(f"Trainning took {time() - t_ini} seconds")