from math import ceil

import torch
from torch import nn

# padding necessary in conv to achieve output of size o with:
#  - input of size i
#  - kernel size k
#  - stride s
pad = lambda i, o, k, s: int(ceil((s*(o-1)-i+k)/s))


class Discriminator (nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self._convolutions = nn.Sequential(
            nn.Conv2d(1,64,4,2,padding=pad(64,32,4,2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(64,128,4,2,padding=pad(64,128,4,2)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(128,256,4,2,padding=pad(128,256,4,2)),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(256,512,4,2,padding=pad(256,512,4,2)),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv2d(512,1,4,1,padding=pad(256,512,4,2)),
            nn.Sigmoid(),
        )
    
    def forward (self, input):
        x = self._convolutions(input)
        return x
    
    @property
    def name (self):
        return "DCGan Discriminator"
    
    @property
    def params (self):
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_params (self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Generator (nn.Module):
    def __init__ (self):
        super(Generator, self).__init__()
        self._deconvolutions = nn.Sequential(
            nn.ConvTranspose2d(100,512,4,1,padding=0,output_padding=0),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512,256,4,2,padding=1,output_padding=0),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256,128,4,2,padding=1,output_padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128,64,4,2,padding=1,output_padding=0),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64,1,4,2,padding=1,output_padding=0),
        )
    
    def forward (self, inputs):
        x = inputs.unsqueeze(dim=2).unsqueeze(dim=2)
        x = self._deconvolutions(x)
        return x
    
    @property
    def input_size (self):
        return 100
    
        
    @property
    def name (self):
        return "DCGan Generator"
    
    @property
    def params (self):
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_params (self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)