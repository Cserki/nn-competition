# -*- coding: utf-8 -*-
"""pretrained_nn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jBG_n8X50-qGDKwnraQl_zOOCNTXePED

#Imports
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from google.colab import drive
drive.mount('/content/drive')

"""#Load Data"""

img_size = 150
transform_train = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    #transforms.CenterCrop((224,224)),
    transforms.ToTensor()
])

train_set = torchvision.datasets.ImageFolder(root = '/content/drive/My Drive/NN_2020_Kaggle_dataset/train',
                                    transform = transform_train)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=2)
mean = 0.
std = 0.
for images, _ in train_loader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(train_loader.dataset)
std /= len(train_loader.dataset)
print(mean)
print(std)

img_size = 150
#img_size = 256
normalize = transforms.Normalize(mean,std)
transform_train = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    #transforms.CenterCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    #transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    normalize
])

#------------------Train stuff------------------
train_set = torchvision.datasets.ImageFolder(root = '/content/drive/My Drive/NN_2020_Kaggle_dataset/train', 
                                             transform = transform_train)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=2)

#------------------Test stuff------------------
test_set = torchvision.datasets.ImageFolder(root = '/content/drive/My Drive/NN_2020_Kaggle_dataset/test', 
                                            transform = transform_test)

test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)

"""#Init network"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""VGG19 150x150 img_size"""

model = torchvision.models.vgg19(pretrained = True)
for p in model.parameters() : 
    p.requires_grad = False
model.classifier = nn.Sequential(
  nn.Linear(in_features=25088, out_features=2048) ,
  nn.ReLU(),
  nn.Linear(in_features=2048, out_features=512) ,
  nn.ReLU(),
  nn.Dropout(p=0.6), 
  nn.Linear(in_features=512 , out_features=16),
  nn.LogSoftmax(dim=1)  
)
model.to(device)

"""VGG16 im_size 256x256"""

model = torchvision.models.vgg16(pretrained=True)
for p in model.parameters() : 
    p.requires_grad = False
model.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=16, bias=True)
  )
model.to(device)

model = torchvision.models.resnet50(pretrained = True)
for p in model.parameters() : 
    p.requires_grad = False
model.fc = nn.Sequential(
  nn.Linear(in_features=2048, out_features=1024) ,
  nn.ReLU(),
  nn.Linear(in_features=1024, out_features=512) ,
  nn.ReLU(),
  nn.Dropout(p=0.6), 
  nn.Linear(in_features=512 , out_features=16),
  nn.LogSoftmax(dim=1)  
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.00001)

losses = []
acc_list = []

def train_for_one_epoch(step):
  print("Training...")
  train_loss = 0
  train_correct = 0
  total = 0

  for batch_num, (data, target) in enumerate(train_loader):
    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    prediction = torch.max(output, 1)

    total += target.size(0)

    train_correct += sum((prediction[1] == target)).item()

    lo = train_loss / (batch_num + 1)
    acc = train_correct / total * 100

    losses.append(lo)
    acc_list.append(acc) 

    print("Step: " + str(step))
    print("Loss: " + str(lo) + " Accuracy: " + str(acc))
  return train_loss, acc

for i in range(30):
  results = train_for_one_epoch(i)

"""#Write Out The Predictions"""

from os import listdir
from os.path import isfile, join
file_names = [f for f in listdir('/content/drive/My Drive/NN_2020_Kaggle_dataset/test/test') if isfile(join('/content/drive/My Drive/NN_2020_Kaggle_dataset/test/test', f))]
print(len(file_names))
file_names.sort()
print(file_names)

output_file = open("last_try.csv", "w")
output_file.write("Id,Category"+"\n")
j = 0
pred = []
with torch.no_grad():
  for data in test_loader:
    images, _ = data
    images = images.to(device)
    model.eval()
    outputs = model(images)
    predictions = torch.max(outputs.data, 1)

    for i in predictions[1]:
      output_file.write(str(file_names[j])+","+str(i.detach().cpu().numpy())+"\n")
      j+=1