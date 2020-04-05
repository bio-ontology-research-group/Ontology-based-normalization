import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from random import randint 
import numpy as np
import time
import os 
import sys 
import numpy
import sklearn 

#Hyperparameters
num_epochs = 100
num_classes = 2
batch_size = 50
learning_rate = 0.0001

#MODEL_STORE_PATH = "/home/smailif/Desktop/Academics/Fall_2019/Self-Learning on Onologies/CHemical_Disease_Updated"

#Load dataset 
X_train_1= numpy.loadtxt("X_train_1")
X_train_2= numpy.loadtxt("X_train_2")
y_train= numpy.loadtxt("Y_train")

X_test_1= numpy.loadtxt("X_test_1")
X_test_2= numpy.loadtxt("X_test_2")
y_test= numpy.loadtxt("Y_test")

#transform to torch
train_x1= torch.from_numpy(X_train_1).float()
train_x2= torch.from_numpy(X_train_2).float()
train_x = [train_x1, train_x2]
train_label= torch.from_numpy(y_train).long()


test_x1 = torch.from_numpy(X_test_1).float()
test_x2 = torch.from_numpy(X_test_2).float()
test_x=[test_x1, test_x2]
test_label= torch.from_numpy(y_test).long()


train_data = []
train_data.append([train_x, train_label])

test_data = []
test_data.append([test_x,test_label])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#Define Network 
class Net (nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Linear (200, 600),
			nn.ReLU())
		self.layer2 = nn.Sequential (
			nn.Linear (600,400),
			nn.ReLU())
		self.layer3 = nn.Sequential(
			nn.Linear (400, 200),
			nn.ReLU())
		self.drop_out = nn.Dropout()
		self.dis = nn.Linear (200,2)

				
	def forward (self, data):
		res = []
		for i in range(2):
			x = data[i]
			out = self.layer1(x)
			out = self.layer2(out)
			out = self.layer3(out)
			out = self.drop_out(out)
			#out = out.reshape(out.size(0),-1)
			res.append(out)
		output = torch.abs(res[1] - res[0])
		#output = torch.mm(res[1] , res[0])		
		output = self.dis(output)
		return output

#Create network 
network = Net()

# Use Cross Entropy for back propagation 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam (network.parameters(),lr=learning_rate)

# Train the model 
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range (num_epochs):
	for i, (train_x, train_label) in enumerate (train_loader):
		# Get data
		inputs = train_x
		labels = train_label

		# Run the forward pass
		outputs = network (inputs)
		outputs=outputs.reshape(-1,2)
		labels=labels.reshape(-1)				
		#print (outputs.size())
		#print (labels.size())
		loss = criterion (outputs, labels)
		loss_list.append(loss.item())
	
		# Back propagation and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Get prediction
		total = labels.size(0)
		_,predicted = torch.max(outputs.data,1)
		correct = (predicted == labels).sum().item()
		acc_list.append (correct/total)
		
		#if (i + 1) % 100 == 0:
		print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct/total)*100))


# Test the model 
network.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for test_x,test_label in  test_loader:
		outputs = network (test_x)
		labels = test_label
		outputs=outputs.reshape(-1,2)		
		array = outputs.data.cpu().numpy()
		numpy.savetxt('output.csv',array)
		labels=labels.reshape(-1)	
		_, predicted = torch.max(outputs.data,1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	print ('Accuracy of model on test dataset is: {} %'.format((correct / total) *100))





















