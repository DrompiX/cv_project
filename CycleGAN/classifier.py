import pickle

import torch
import torch.nn as nn
import torchvision.transforms as T
import os 

from PIL import Image

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FruitClassifier(nn.Module):

    def __init__(self, classes):
        super(FruitClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 64, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(5),
            Flatten(),
            nn.Linear(64, 100),
            nn.ReLU(),
            nn.Linear(100, classes)
        )
        with open('id2class.pkl', 'rb') as fd:
            id2class = pickle.load(fd)
        self.id2class = id2class

    def forward(self, x):
        return self.model(x)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, x):
        return self.id2class[self.model(x).max(1)[1].cpu().item()]


def load_image(filename, device='cpu'):
    transform = T.Compose([T.ToTensor(),
                           T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = Image.open(filename)
    img_t = transform(img)
    return img_t.unsqueeze(0).to(device)



def predictFolder(path):
    predictions = []
    clf = FruitClassifier(5)
    clf.load_weights('fruit_classifier.pth')
    for img_path in os.listdir(path):
        img = load_image(path+"/"+img_path)
        predictions.append(clf.predict(img))
    return predictions

