from math import ceil

import torch.nn as nn
import torch

# padding necessary in conv to achieve output of size o with:
#  - input of size i
#  - kernel size k
#  - stride s
pad = lambda i, o, k, s: int(ceil((s*(o-1)-i+k)/s))


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