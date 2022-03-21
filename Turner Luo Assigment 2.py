#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms


# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# Question A:\
# Model 0: CNN\
# Model 1: DNN with 0 hidden layer\
# Model 2: DNN with 1 hidden layer\
# Model 3: DNN with 2 hidden layer\
# Model 4: DNN with 3 hidden layer\
# Model 5: DNN with 4 hidden layer\
# 
# Model 0:

# In[4]:


import torch.nn as nn
import torch.nn.functional as F


class CNN0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model0 = CNN0()


# Model 1:
# 

# In[5]:


class DNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3,10)
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        return x
model1 = DNN1()


# Model 2:

# In[6]:


class DNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3,512)
        self.fc2 = nn.Linear(512,10)
    def forward(self,x):
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        self.dropout = nn.Dropout(0.5)
        x = self.fc2(x)
        return x 
model2 = DNN2()


# Model 3:

# In[7]:


class DNN3(nn.Module):
   def __init__(self):
       super().__init__()
       self.fc1 = nn.Linear(32*32*3,512)
       self.fc2 = nn.Linear(512,512)
       self.fc3 = nn.Linear(512,10)
   def forward(self,x):
       x = torch.flatten(x,1)
       x = F.relu(self.fc1(x))
       self.dropout = nn.Dropout(0.5)
       x = self.fc2(x)
       self.dropout = nn.Dropout(0.5)
       x = self.fc3(x)
       return x
model3 = DNN3()


# Model 4:

# In[8]:


class DNN4(nn.Module):
   def __init__(self):
       super().__init__()
       self.fc1 = nn.Linear(32*32*3,512)
       self.fc2 = nn.Linear(512,512)
       self.fc3 = nn.Linear(512,512)
       self.fc4 = nn.Linear(512,10)
   def forward(self,x):
       x = torch.flatten(x,1)
       x = F.relu(self.fc1(x))
       self.dropout = nn.Dropout(0.5)
       x = self.fc2(x)
       self.dropout = nn.Dropout(0.5)
       x = self.fc3(x)
       self.dropout = nn.Dropout(0.5)
       x = self.fc4(x)
       return x
model4 = DNN4()


# Model 5:

# In[9]:


class DNN5(nn.Module):
   def __init__(self):
       super().__init__()
       self.fc1 = nn.Linear(32*32*3,512)
       self.fc2 = nn.Linear(512,512)
       self.fc3 = nn.Linear(512,512)
       self.fc4 = nn.Linear(512,512)
       self.fc5 = nn.Linear(512,10)
   def forward(self,x):
       x = torch.flatten(x,1)
       x = F.relu(self.fc1(x))
       self.dropout = nn.Dropout(0.5)
       x = self.fc2(x)
       self.dropout = nn.Dropout(0.5)
       x = self.fc3(x)
       self.dropout = nn.Dropout(0.5)
       x = self.fc4(x)
       self.dropout = nn.Dropout(0.5)
       x = self.fc5(x)
       return x
model5 = DNN5()


# Training the model:\
# Setting the optimizers for each models:\
# Define accuracy function:\
# Return loss and accuracy:

# In[11]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer0 = optim.SGD(model0.parameters(), lr=0.001, momentum=0.9)
optimizer1 = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
optimizer3 = optim.SGD(model3.parameters(), lr=0.001, momentum=0.9)
optimizer4 = optim.SGD(model4.parameters(), lr=0.001, momentum=0.9)
optimizer5 = optim.SGD(model5.parameters(), lr=0.001, momentum=0.9)


# Model 0:

# In[12]:


accuracy0 = []
for epoch in range(10):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer0.zero_grad()
        outputs = model0(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer0.step()
        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
       for data in testloader:
          images, labels = data
          outputs = model0(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    ac = correct / total
    accuracy0.append(ac)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('Finished Training')


# Model 1:

# In[13]:


accuracy1 = []
for epoch in range(10):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer1.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer1.step()
        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
       for data in testloader:
          images, labels = data
          outputs = model1(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    ac = correct / total
    accuracy1.append(ac)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('Finished Training')


# Model 2:

# In[14]:


accuracy2 = []
for epoch in range(10):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()
        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
       for data in testloader:
          images, labels = data
          outputs = model2(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    ac = correct / total
    accuracy2.append(ac)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('Finished Training')


# Model 3:

# In[15]:


accuracy3 = []
for epoch in range(10):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer3.zero_grad()
        outputs = model3(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer3.step()
        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
       for data in testloader:
          images, labels = data
          outputs = model3(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    ac = correct / total
    accuracy3.append(ac)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('Finished Training')


# Model 4:

# In[16]:


accuracy4 = []
for epoch in range(10):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer4.zero_grad()
        outputs = model4(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer4.step()
        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
       for data in testloader:
          images, labels = data
          outputs = model4(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    ac = correct / total
    accuracy4.append(ac)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('Finished Training')


# Model 5:

# In[17]:


accuracy5 = []
for epoch in range(10):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer5.zero_grad()
        outputs = model5(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer5.step()
        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
       for data in testloader:
          images, labels = data
          outputs = model5(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    ac = correct / total
    accuracy5.append(ac)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('Finished Training')


# In[19]:


e = [1,2,3,4,5,6,7,8,9,10]
plt.plot(e, accuracy0, label = 'CNN')
plt.plot(e, accuracy1, label = 'Simple Dense -- 0 Hidden')
plt.plot(e, accuracy2, label = 'Simple Dense -- 1 Hidden')
plt.plot(e, accuracy3, label = 'Simple Dense -- 2 Hidden')
plt.plot(e, accuracy4, label = 'Simple Dense -- 3 Hidden')
plt.plot(e, accuracy5, label = 'Simple Dense -- 4 Hidden')


plt.title("Test Accuracy of Various Models on CIFAR-10")
plt.legend(loc="upper left")
plt.show()


# The CNN has the best result. It is easier for optimizer to find better weights with CNN. We can see, with multiple hidden layers, the accuracy did not significantly improve. Maybe optimizer failed to find the best weights.

# Question B:\
# CNN model with Sigmoid function (Model 6):
# 

# In[20]:


class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.sigmoid(self.conv1(x)))
        x = self.pool(torch.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


model6 = CNN1()


# In[21]:


accuracy6 = []
optimizer6 = optim.SGD(model6.parameters(), lr=0.001, momentum=0.9)
for epoch in range(10):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer6.zero_grad()
        outputs = model6(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer6.step()
        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
       for data in testloader:
          images, labels = data
          outputs = model6(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    ac = correct / total
    accuracy6.append(ac)
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

print('Finished Training')


# In[25]:


plt.plot(accuracy0,label='CNN -- ReLU')
plt.plot(accuracy6,label='CNN -- Sigmoid')
plt.title('Test Accuracy of CNN on CIFAR-10 using ReLU vs Sigmoid Activation')
plt.legend(loc="upper left")
plt.show()


# ReLU activation function gives better result. Since sigmoid function are better with gradient vanishing problems.

# Question C:\
# Model with drop out (Model 7):

# In[27]:


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        self.dropout = nn.dropout(0.5)
        x = F.relu(self.fc2(x))
        self.dropout = nn.dropout(0.5)
        x = self.fc3(x)
        return x


model7 = CNN2()
optimizer7 = optim.SGD(model7.parameters(), lr=0.001, momentum=0.9)


# In[ ]:




