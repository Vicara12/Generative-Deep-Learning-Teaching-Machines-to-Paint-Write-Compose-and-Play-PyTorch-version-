import torchvision as tv
from torchvision import transforms as tr
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
import torch
from PIL import Image
from urllib.request import urlretrieve
import os
from torchvision.transforms.v2 import Lambda
from math import prod

use_cnn = True
input_res = 32

batch_size = 64
labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
model_name = f"model_{'cnn' if use_cnn else "mlp"}"

to_one_hot = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

img_transf = tr.Compose([tr.Resize((input_res,input_res)),
                         tr.ToTensor()])

train_data = tv.datasets.CIFAR10(os.path.realpath(__file__),
                                 train = True,
                                 transform=img_transf,
                                 target_transform=to_one_hot,
                                 download=True)
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = tv.datasets.CIFAR10(os.path.realpath(__file__),
                                train = False,
                                transform=img_transf,
                                target_transform=to_one_hot,
                                download=True)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

class MLP (nn.Module):
  def __init__ (self, dims_in, dims_out):
    super(MLP, self).__init__()
    self.inp = prod(dims_in)
    self.fc1 = nn.Linear(self.inp, 200)
    self.fc2 = nn.Linear(200, 150)
    self.fc3 = nn.Linear(150, dims_out)
  
  def forward (self, x):
    x = x.view(-1, self.inp)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
  @property
  def params (self):
    return sum(p.numel() for p in self.parameters())

  @property
  def trainable_params (self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
  @property
  def name (self):
    return "MLP"

class CNN (nn.Module):
  def __init__ (self):
    super(CNN, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),

      nn.Conv2d(32, 32, kernel_size=3, stride=2),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),

      nn.Conv2d(32, 64, kernel_size=3, stride=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),

      nn.Conv2d(64, 64, kernel_size=3, stride=2),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
    )
    self.classifier = nn.Sequential(
      nn.Linear(64*25, 128),
      nn.BatchNorm1d(1),
      nn.LeakyReLU(),
      nn.Dropout(),
      nn.Linear(128, 10)
    )
  
  def forward (self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = x.unsqueeze(1)
    x = self.classifier(x)
    return torch.flatten(x, 1)
  
  @property
  def params (self):
    return sum(p.numel() for p in self.parameters())

  @property
  def trainable_params (self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
  @property
  def name (self):
    return "CNN"

if use_cnn:
  net = CNN()
else:
  net = MLP((3,input_res,input_res), len(labels))
if input("load model?: ")[0] == 'y':
  net_state_dict = torch.load(os.path.realpath(__file__)+f"model/{model_name}")
  net.load_state_dict(net_state_dict)
  
optimizer = optim.Adam(net.parameters(), lr=0.0005)

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
  t_ini = time()
  for epoch in range(epochs):
    training_loss = 0.0
    valid_loss = 0.0
    model.train()
    for batch in train_loader:
      optimizer.zero_grad()
      inputs, target = batch
      inputs = inputs.to(device)
      target = target.to(device)
      output = model(inputs)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      training_loss += loss.data.item() * inputs.size(0)
    training_loss /= len(train_loader.dataset)
    model.eval()
    num_correct = 0
    num_examples = 0
    for batch in val_loader:
      inputs, targets = batch
      inputs = inputs.to(device)
      output = model(inputs)
      targets = targets.to(device)
      loss = loss_fn(output, targets)
      valid_loss += loss.data.item() * inputs.size(0)
      correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], torch.max(targets, dim=1)[1]).view(-1)
      num_correct += torch.sum(correct).item()
      num_examples += correct.shape[0]
    valid_loss /= len(val_loader.dataset)
    print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))
  print(f"Trainning took {time() - t_ini} seconds")

def predict (image, img_trans, labels):
  img = Image.open(image)
  img = img_trans(img).unsqueeze(0).to(device)
  out = F.softmax(net(img))
  prediction = torch.argmax(out).item()
  return labels[prediction], out


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
        torch.nn.CrossEntropyLoss(),
        train_data_loader,
        train_data_loader,
        epochs=100,
        device=device)

if input("save model?: ")[0] == 'y':
  torch.save(net.state_dict(), os.path.realpath(__file__)+f"model/{model_name}")

while True:
  url = input("image url: ")
  try:
    urlretrieve(url, "image")
  except ValueError:
    print("ERROR: could not load image")
  label, results = predict("image", img_transf, labels)
  print(f"result -> {label}")
  for i in range(len(labels)):
    print(f" - {labels[i]}: {round(results[0,i].item()*100,2)} %")
  os.remove("image")