from tools import *
from models import *

from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

class SignLangDetector():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.model = M2()
        self.model.load_state_dict(torch.load('./checkpoints/model95.pkl'))
        self.prediction = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't' , 'u', 'v', 'w', 'x', 'y']


    def predict(self, img):
        img  = Image.open(img)
        data = self.transform(img)
        data = data.unsqueeze(dim=0)
        pred = self.model.forward(data)
        pred = pred.max(1)[1]
        print(pred)
        # print('Prediction: ', self.prediction[pred[0]])
        return self.prediction[pred[0]]
