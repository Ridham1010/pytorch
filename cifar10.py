!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c cifar-10

from google.colab import drive
drive.mount('/content/drive')

!ls

from zipfile import ZipFile
dataset = '/content/cifar-10.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('Done')

!pip install py7zr

import py7zr
archive = py7zr.SevenZipFile('/content/train.7z', mode='r')
archive.extractall()#if you want to extract to a specific folder you can specif path=' '
archive.close()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split


filenames=os.listdir('/content/train')
len(filenames)

labels_df=pd.read_csv('/content/trainLabels.csv')
labels_df.head()

classtoidx={name:label for label,name in enumerate(sorted(labels_df['label'].unique()))}

labels = np.array(labels_df['label'].map(classtoidx))

id_list=list(labels_df['id'])

train_folder='/content/train'
data=[]
for id in id_list:
  img=train_folder+'/'+str(id)+'.png'
  data.append(img)

img = plt.imread(data[0])
print(img.shape)


from PIL import Image

X = []
for path in data:
    img = Image.open(path)
    img = img.convert('RGB')  # ensures 3 channels
    X.append(np.array(img))

X = np.array(X)
print(X.shape)  # Should now print (50000, 32, 32, 3)
Y = np.array(labels)
print(Y.shape)  # (50000,)


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

X_train=X_train/255

X_test=X_test/255

X_train

!pip install torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray((image * 255).astype(np.uint8))  # convert back to PIL
        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ResNet50 needs 224x224
    transforms.ToTensor()
])

train_data = CIFAR10Dataset(X_train, Y_train, transform=transform)
test_data = CIFAR10Dataset(X_test, Y_test, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 5                   # WARNING: This is going to take a lot of time approximately 35 minutes.....

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


torch.save(model.state_dict(), "/content/drive/MyDrive/resnet50_cifar10.pth")
# to load the model
# # 1. Mount Drive again
# from google.colab import drive
# drive.mount('/content/drive')

# # 2. Load model architecture (must match the saved one)
# import torchvision.models as models
# import torch.nn as nn
# import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = models.resnet50(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
# model.load_state_dict(torch.load("/content/drive/MyDrive/resnet50_cifar10.pth"))
# model.to(device)
# model.eval()


