import os

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision as tv
from torchvision import transforms as tr

from models import Encoder, Decoder, Autoencoder
from trainer import train


loss_fn = torch.nn.BCELoss()
# loss_fn = torch.nn.MSELoss()
batch_size = 64

img_transf = tr.Compose([tr.Resize((32,32)),
                         tr.ToTensor()])

dataset = tv.datasets.FashionMNIST

dirname = os.path.dirname(os.path.realpath(__file__))

train_data = dataset(dirname,
                     train = True,
                     transform=img_transf,
                     download=True)
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = dataset(dirname,
                    train = False,
                    transform=img_transf,
                    download=True)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

net = Autoencoder(2)

if input("load model?: ")[0] == 'y':
  net_state_dict = torch.load(dirname+f"model/autoencoder")
  net.load_state_dict(net_state_dict)

optimizer = optim.Adam(net.parameters(), lr=0.0005)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print(f"using device {device}")
net.to(device)

print(f"INFO: model {net.name} has {net.params} params ({net.trainable_params} trainable)")

if input("train model?: ")[0] == 'y':
  train(net,
        optimizer,
        loss_fn,
        train_data_loader,
        train_data_loader,
        epochs=100,
        device=device)

if input("save model?: ")[0] == 'y':
  torch.save(net.state_dict(), dirname+f"model/autoencoder")