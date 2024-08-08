from math import ceil

import torch.nn as nn
import torch

# padding necessary in conv to achieve output of size o with:
#  - input of size i
#  - kernel size k
#  - stride s
pad = lambda i, o, k, s: int(ceil((s*(o-1)-i+k)/s))

# ------------- autoencoder -------------

class Encoder (nn.Module):
  def __init__ (self, intermediate_size):
    super(Encoder, self).__init__()
    self._convolutions = nn.Sequential(
      nn.Conv2d(1, 32, 3, 2, padding=pad(32, 16, 3, 2)),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, 2, padding=pad(16, 8, 3, 2)),
      nn.ReLU(),
      nn.Conv2d(64, 128, 3, 2, padding=pad(8, 4, 3, 2)),
      nn.ReLU()
    )
    self._dense = nn.Linear(4*4*128, intermediate_size)
  
  def forward (self, x):
    x = self._convolutions(x)
    x = torch.flatten(x, 1)
    x = self._dense(x)
    return x
  
  @property
  def name (self):
    return "Encoder"

  @property
  def params (self):
    return sum(p.numel() for p in self.parameters())

  @property
  def trainable_params (self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Decoder (nn.Module):
  def __init__ (self, intermediate_size):
    super(Decoder, self).__init__()
    self._intermediate_size = intermediate_size
    self._dense = nn.Linear(intermediate_size, 2048)
    self._convs = nn.Sequential(
      # padding has to be carefully selected to achieve sizes 8 -> 16 -> 32
      nn.ConvTranspose2d(128, 128, 3, 2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 1, 3, 1, padding=1)
    )

  
  def forward (self, x):
    x = self._dense(x)
    x = torch.reshape(x, (x.shape[0],128,4,4))
    x = self._convs(x)
    return torch.sigmoid(x)
  
  @property
  def name (self):
    return "Decoder"
  
  @property
  def params (self):
    return sum(p.numel() for p in self.parameters())

  @property
  def trainable_params (self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Autoencoder (nn.Module):
  def __init__ (self, intermediate_size):
    super(Autoencoder, self).__init__()
    self._encoder = Encoder(intermediate_size)
    self._decoder = Decoder(intermediate_size)
  
  def forward (self, x):
    x = self._encoder(x)
    x = self._decoder(x)
    return x
  
  @property
  def encoder (self):
    return self._encoder
  
  @property
  def decoder (self):
    return self._decoder
  
  @property
  def name (self):
    return "Autoencoder"
  
  @property
  def params (self):
    return self._encoder.params + self._decoder.params

  @property
  def trainable_params (self):
    return self._encoder.trainable_params + self._decoder.trainable_params


# ------------- variational autoencoder -------------

class Sampling (nn.Module):
  def __init__ (self, device):
    super(Sampling, self).__init__()
    self._device = device
  
  def forward (self, input):
    z_mean, z_log_var = input
    mean = torch.zeros(z_mean.shape[0:2])
    std = torch.ones(z_mean.shape[0:2])
    epsilon = torch.normal(mean=mean, std=std).to(self._device)
    return z_mean + torch.exp(0.5*z_log_var) * epsilon

class VariationalEncoder (nn.Module):
  def __init__ (self, intermediate_size, device):
    super(VariationalEncoder, self).__init__()
    self._convolutions = nn.Sequential(
      nn.Conv2d(1, 32, 3, 2, padding=pad(32, 16, 3, 2)),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, 2, padding=pad(16, 8, 3, 2)),
      nn.ReLU(),
      nn.Conv2d(64, 128, 3, 2, padding=pad(8, 4, 3, 2)),
      nn.ReLU()
    )
    self._dense_mean = nn.Linear(4*4*128, intermediate_size)
    self._dense_log_var = nn.Linear(4*4*128, intermediate_size)
    self._sampler = Sampling(device)
  
  def forward (self, x, stats=False):
    x = self._convolutions(x)
    x = torch.flatten(x, 1)
    z_mean = self._dense_mean(x)
    z_log_var = self._dense_log_var(x)
    z = self._sampler([z_mean, z_log_var])
    if stats:
      return z_mean, z_log_var, z
    else:
      return z
  
  @property
  def name (self):
    return "Variational Encoder"

  @property
  def params (self):
    return sum(p.numel() for p in self.parameters())

  @property
  def trainable_params (self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VariationalAutoencoder (nn.Module):
  def __init__ (self, intermediate_size, device):
    super(VariationalAutoencoder, self).__init__()
    self._encoder = VariationalEncoder(intermediate_size, device)
    self._decoder = Decoder(intermediate_size)
  
  def forward (self, x, stats=False):
    if stats:
      z_mean, z_log_var, z = self._encoder(x, True)
    else:
      z = self._encoder(x)
    x = self._decoder(z)
    if stats:
      return z_mean, z_log_var, x
    else:
      return x
  
  @property
  def encoder (self):
    return self._encoder
  
  @property
  def decoder (self):
    return self._decoder
  
  @property
  def name (self):
    return "Variational Autoencoder"
  
  @property
  def params (self):
    return self._encoder.params + self._decoder.params

  @property
  def trainable_params (self):
    return self._encoder.trainable_params + self._decoder.trainable_params