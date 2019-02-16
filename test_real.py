import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from tools import *
from models import *
## Methods

transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

hand_test_data = torchvision.datasets.ImageFolder('../dataset/data/new_test/', transform = transform)
test_dataloader = torch.utils.data.DataLoader(hand_test_data,
                                          batch_size=1,
                                          shuffle=False)

model = M2()
model.load_state_dict(torch.load('./checkpoints/model45.pkl'))
# model = model.cuda()

prediction = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't' , 'u', 'v', 'x', 'y']

# train(model, train_dataloader, test_dataloader, 100)


for _, sample in enumerate(test_dataloader):
    data = sample[0] #.cuda()
    label = torch.tensor(sample[1]) #.cuda()
    
    pred = model.forward(data)
    pred = pred.max(1)[1]
    print('Prediction: ', prediction[pred[0][0]])

