import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import cv2


class_id_pairs = {"Algae.Bacillariophyta.Asterionella":0,
                  "Algae.Bacillariophyta.Aulacoseira":1,
                  "Algae.Bacillariophyta.Centrales":2,
                  "Algae.Bacillariophyta.Nitzschia_acicularis":3,
                  "Algae.Bacillariophyta.Nitzschia_palea":4,
                  "Algae.Chlorophyta.Coelastrum":5,
                  "Algae.Chlorophyta.Monoraphidium":6,
                  "Algae.Chlorophyta.Pediastrum":7,
                  "Algae.Chlorophyta.Scenedesmus":8,
                  "Algae.Chlorophyta.Schroederia":9,
                  "Algae.Chlorophyta.Tetraedron":10,
                  "Algae.Chlorophyta.Tetrastrum":11,
                  "Bacteria.Cyanobacteria.Anabaena_solitaria":12,
                  "Bacteria.Cyanobacteria.Cyanobacteria_coenobia":13,
                  "Bacteria.Cyanobacteria.Filamentous":14,
                  "Bacteria.Cyanobacteria.Spirulina":15}

directory_cserki_train = '/home/cserki/PycharmProjects/nn_competition/nn/NN_2020_Kaggle_dataset/train_data'
directory_cserki_test = '/home/cserki/PycharmProjects/nn_competition/nn/NN_2020_Kaggle_dataset'


'''
training_set = []

im_size = 50
train_transforms = transforms.Compose([
 transforms.Resize((im_size,im_size)),
 transforms.RandomRotation(0), transforms.RandomHorizontalFlip(),
 transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


train_data = torchvision.datasets.ImageFolder(directory_cserki_training, train_transforms)




data = []
for category,id in class_id_pairs.items():
    path = os.path.join(directory_cserki_training, category)
    class_num = id
    for images in os.listdir(path):
        print(images)
        img_array = cv2.imread(os.path.join(path, images), cv2.IMREAD_COLOR)
        new = cv2.resize(img_array,(im_size,im_size))
        plt.imshow(new)
        plt.show()
        #print(str(img_array.shape[0]) + 'x' + str(img_array.shape[1]))
        break
    break
#print(img_array)
print(train_data[412])
'''
"""# Load Data"""

im_size = 32
train_transform = transforms.Compose([transforms.Resize((im_size,im_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.Compose([transforms.Resize((im_size,im_size)), transforms.ToTensor()])

#------------------Train stuff------------------
train_set = torchvision.datasets.ImageFolder(directory_cserki_train, train_transform)
train_loader = torch.utils.data.DataLoader(train_set, 128, True)

#------------------Test stuff------------------
test_set = torchvision.datasets.ImageFolder(directory_cserki_test, test_transform)
test_loader = torch.utils.data.DataLoader(test_set, 128, False)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

import matplotlib.pyplot as plt
plt.imshow(example_data[0][0])

print(example_targets)

"""#Define Network"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class AlexNet(nn.Module):
  def __init__(self, num_classes = 16):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(kernel_size = 2),
        nn.Conv2d(64, 192, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(kernel_size = 2),
        nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(kernel_size = 2),
        nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True)
    )
    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256*2*2, 4096),
        nn.ReLU(inplace = True),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace = True),
        nn.Linear(4096, 16)
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256*2*2)
    x = self.classifier(x)
    return x

net = AlexNet()
print(net)

net = net.to(device)

"""#Define Optimizer"""

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), 0.02, momentum = 0.9)

"""#Train Network"""

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
    output = net(data)
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

for i in range(2):
  results = train_for_one_epoch(i)

"""#Test"""

correct = 0
total = 0

output_file = open('result.csv','a')
output_file.write("Id,Category\n")

with torch.no_grad():
  for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)

    outputs = net(images)

    predictions = torch.max(outputs.data, 1)
    output_file.write(str(images) + "," + str(predictions[1]) + "\n")
    total += labels.size(0)
    correct += sum((predictions[1]==labels)).item()
  print("Accuracy is: " + str(100*correct/total))
  output_file.close()

