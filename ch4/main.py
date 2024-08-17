import os
from itertools import chain

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision as tv
from torchvision import transforms as tr

from models import Generator, Discriminator
from trainer import train

wgan = False
loss_fn = torch.nn.BCELoss()
# loss_fn = torch.nn.MSELoss()
batch_size = 64

model_name = "dcgan"

img_transf = tr.Compose([tr.Resize((64,64)),
                         tr.Grayscale(),
                         tr.ToTensor()])


dirname = os.path.dirname(os.path.realpath(__file__))+"/.."

# Get dataset from: https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images?resource=download
# Put folder with images as "brick_dataset" in the dataset folder

train_data = tv.datasets.ImageFolder(dirname+"/datasets/brick_dataset", transform=img_transf)
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print(f"using device {device}")

if wgan:
  raise Exception()
else:
  generator = Generator()
  discriminator = Discriminator()
generator.to(device)
discriminator.to(device)

if input("load model?: ")[0] == 'y':
  gen_state_dict = torch.load(dirname+f"/models/{model_name}_gen")
  disc_state_dict = torch.load(dirname+f"/models/{model_name}_disc")
  generator.load_state_dict(gen_state_dict)
  discriminator.load_state_dict(disc_state_dict)

optimizer = optim.Adam(chain(generator.parameters(), discriminator.parameters()), lr=0.001)

# print(f"INFO: model {generator.name} has {generator.params} params ({generator.trainable_params} trainable)")
# print(f"INFO: model {discriminator.name} has {discriminator.params} params ({discriminator.trainable_params} trainable)")

train_f = train
train_f(generator, discriminator, optimizer, loss_fn, train_data_loader,
        ratio=5, loops=5, iterations=1, batch_size=batch_size)

if input("save model?: ")[0] == 'y':
  torch.save(generator.state_dict(), dirname+f"/models/{model_name}_gen")
  torch.save(generator.state_dict(), dirname+f"/models/{model_name}_dis")