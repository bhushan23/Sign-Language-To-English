import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


## Methods

def train(model, train_dataloader, num_epochs = 10):
  optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
  criterion = nn.CrossEntropyLoss()
  
  train_acc = []
  test_acc  = []
  loss_plt  = []
  for i in range(0, num_epochs):
    total_loss = 0
    for bidx, sample in enumerate(train_dataloader):
      data  = sample[0]
      label = sample[1]
      data  = data.cuda()
      label = label.cuda()
      pred  = model.forward(data)
      # print(pred.shape, label.shape)
      # print(pred, label)
      loss  = criterion(pred, label)
      total_loss += loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
    if i % 5 == 0:
      torch.save(model.state_dict(), './checkpoints/model.pkl')
      # test_acc.append(test(model, val_dataloader))
      train_acc.append(test(model, train_dataloader))
    loss_plt.append(total_loss.item())
    print(total_loss.item())  
    
  # plt.plot(train_acc)
  # plt.xlabel('Epoch')
  # plt.ylabel('Training Accuracy')
  # plt.legend()
  # plt.show()
  # plt.plot(loss_plt)
  # plt.xlabel('Epoch')
  # plt.ylabel('Loss')
  # plt.legend()
  # plt.show()
  print('Train Acc: ', train_acc)
  return train_acc, test_acc, loss_plt

## Training


transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

hand_data = torchvision.datasets.ImageFolder('./dataset/train/', transform = transform)
train_dataloader = torch.utils.data.DataLoader(hand_data,
                                          batch_size=64,
                                          shuffle=True)

model = M1()
model = model.cuda()

train(model, train_dataloader)

