import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from random import randint
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import time
import os
import sys
import numpy
import sklearn

# Hyperparameters
num_epochs = 25
num_classes = 2
batch_size = 50
learning_rate = 0.0001

MODEL_STORE_PATH = "/home/smailif/Desktop/Academics/Fall_2019/Self-Learning on Onologies/CHemical_Disease_Updated"
# Load dataset 
X_train= numpy.loadtxt("X_train")
y_train= numpy.loadtxt("Y_train")
X_test= numpy.loadtxt("X_test")
y_test= numpy.loadtxt("Y_test")

train_x= torch.from_numpy(X_train).float()
train_label= torch.from_numpy(y_train).long()
test_x= torch.from_numpy(X_test).float()
test_label= torch.from_numpy(y_test).long()

# Reshape data into 2D matrices
train_x.resize_((train_x.size(0),1,20,20))
test_x.resize_((test_x.size(0),1,20,20))

train_data = []
for i in range(len(train_x)):
	train_data.append([train_x[i], train_label[i]])

test_data = []
for i in range(len(test_x)):
	test_data.append([test_x[i], test_label[i]])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

 
# Define network
class Net (nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), 
			nn.ReLU(), 
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), 
			nn.ReLU(), 
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.drop_out=nn.Dropout()
		self.fc1 = nn.Linear (5*5*64, 100)
		self.fc2 = nn.Linear (100, 2)

	def forward (self,x):
		out = self.layer1(x)
		out = self.layer2 (out)
		out = out.reshape(out.size(0),-1)
		out = self.drop_out(out)
		out = self.fc1 (out)
		out = self.fc2(out)
		return (out)

# Create network 
network = Net()

# Use Cross Entropy for back propagation
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam (network.parameters(), lr=learning_rate)


# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range (num_epochs):
	for i, (train_x, train_label) in enumerate (train_loader):
		# Create the minibatch
		inputs = train_x		
		labels = train_label
	
		# Run the forward pass
		outputs = network (inputs)
		print (outputs.size())
		print (labels.size())		
		loss = criterion (outputs, labels)
		loss_list.append(loss.item())
 
		# Backpropagation and perform Adam optimization		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		#Calculate accuracy 
		total = labels.size(0)
		_,predicted = torch.max(outputs.data,1)
		correct = (predicted == labels).sum().item()
		acc_list.append(correct/total)

		if (i + 1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct/total)*100))

		
# Test the model 
network.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for test_x,test_label in  test_loader:
		outputs = network (test_x)
		labels = test_label
		_, predicted = torch.max(outputs.data,1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	print ('Accuracy of model on test dataset is: {} %'.format((correct / total) *100))


# Save the model and plot curves
torch.save(network.state_dict(),MODEL_STORE_PATH + 'conv_net_model.ckpt')


p = figure (y_axis_label='Loss', width=850, y_range=(0,1), title = 'Pytorch ConvNet results')
p.extra_y_ranges = {'Accuracy':Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label = 'Accuracy (%)'), 'right')
p.line (np.arange(len(loss_list)), loss_list)
p.line (np.arange(len(loss_list)),np.array(acc_list)*100, y_range_name='Accuracy', color='red')
show(p)





























