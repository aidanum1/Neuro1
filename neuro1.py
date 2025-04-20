# -*- coding: utf-8 -*-


from google.colab import files
files.upload()

import os
import zipfile

os.makedirs("/root/.kaggle", exist_ok=True)


!mv kaggle.json /root/.kaggle/

!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection

!unzip /content/brain-mri-images-for-brain-tumor-detection.zip

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from torch.utils.data import Subset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = "/content/brain_tumor_dataset"
transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
dataset = ImageFolder(root=data_path, transform=transform)
targets = np.array([label for _, label in dataset])
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_val_idx, test_idx in splitter.split(np.zeros(len(targets)), targets):
    targets_train_val = targets[train_val_idx]
    splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
    for train_idx, val_idx in splitter_val.split(np.zeros(len(targets_train_val)), targets_train_val):
        train_indices = train_val_idx[train_idx]
        val_indices = train_val_idx[val_idx]
train_dataset = Subset(dataset, train_indices)
val_dataset   = Subset(dataset, val_indices)
test_dataset  = Subset(dataset, test_idx)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = dataset.classes

def imshow(img):
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
dataiter = iter(train_loader)
images, labels = next(dataiter)

for i in range(4):
    imshow(images[i])
    print(f'Label: {classes[labels[i]]}')

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = BrainTumorCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')

