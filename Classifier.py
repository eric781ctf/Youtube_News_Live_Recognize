#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
# In[3]:

def init(PIC,folder):
	with open('news_to_name.json', 'r') as f:
		news_to_name = json.load(f)

	model = load_checkpoint('checkpoint_v2.pth')
	model.cuda()

	probs, classes = predict(folder + PIC, model)
	#print(probs)
	#print(classes)
	
	if classes[0] == '6':
		#print("word")
		return "word"
	elif classes[0]:
		#print("title")
		return "title"
	

#print(len(news_to_name)) 
#print(news_to_name)


# In[10]:




# Function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
	
	checkpoint = torch.load(filepath)
	
	if checkpoint['arch'] == 'vgg16':
		
		model = models.vgg16(pretrained=True)
		
		for param in model.parameters():
			param.requires_grad = False
	else:
		print("Architecture not recognized.")
	
	model.class_to_idx = checkpoint['class_to_idx']
	
	classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
											('relu', nn.ReLU()),
											('drop', nn.Dropout(p=0.5)),
											('fc2', nn.Linear(5000, 102)),
											('output', nn.LogSoftmax(dim=1))]))

	model.classifier = classifier
	
	model.load_state_dict(checkpoint['model_state_dict'])
	
	return model


#print(model)


# In[6]:




def process_image(image_path):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
		returns an Numpy array
	'''
	
	# Process a PIL image for use in a PyTorch model
	
	pil_image = Image.open(image_path)
	
	# Resize
	if pil_image.size[0] > pil_image.size[1]:
		pil_image.thumbnail((5000, 256))
	else:
		pil_image.thumbnail((256, 5000))
		
	# Crop 
	left_margin = (pil_image.width-224)/2
	bottom_margin = (pil_image.height-224)/2
	right_margin = left_margin + 224
	top_margin = bottom_margin + 224
	
	pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
	
	# Normalize
	np_image = np.array(pil_image)/255
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	np_image = (np_image - mean) / std
	
	# PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
	# Color channel needs to be first; retain the order of the other two dimensions.
	np_image = np_image.transpose((2, 0, 1))
	
	return np_image


# In[7]:

'''
def imshow(image, ax=None, title=None):
	if ax is None:
		fig, ax = plt.subplots()
	
	# PyTorch tensors assume the color channel is the first dimension
	# but matplotlib assumes is the third dimension
	image = image.transpose((1, 2, 0))
	
	# Undo preprocessing
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	image = std * image + mean
	
	if title is not None:
		ax.set_title(title)
	
	# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
	image = np.clip(image, 0, 1)
	
	ax.imshow(image)
	
	return ax

image = process_image('test/P2020-10-08-17_24_55.png')
imshow(image)
'''

# In[11]:


# Implement the code to predict the class from an image file

def predict(image_path, model, topk=2):
	''' Predict the class (or classes) of an image using a trained deep learning model.
	'''
	
	image = process_image(image_path)
	
	# Convert image to PyTorch tensor first
	image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
	#print(image.shape)
	#print(type(image))
	
	# Returns a new tensor with a dimension of size one inserted at the specified position.
	image = image.unsqueeze(0)
	
	output = model.forward(image)
	
	probabilities = torch.exp(output)
	
	# Probabilities and the indices of those probabilities corresponding to the classes
	top_probabilities, top_indices = probabilities.topk(topk)
	
	# Convert to lists
	top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
	top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
	
	# Convert topk_indices to the actual class labels using class_to_idx
	# Invert the dictionary so you get a mapping from index to class.
	
	idx_to_class = {value: key for key, value in model.class_to_idx.items()}
	#print(idx_to_class)
	
	top_classes = [idx_to_class[index] for index in top_indices]
	
	return top_probabilities, top_classes



# In[12]:

'''
# Display an image along with the top 5 classes

# Plot news input image
plt.figure(figsize = (6,10))
plot_1 = plt.subplot(2,1,1)

#image = process_image('flowers/test/1/image_06743.jpg')
image = process_image('test/P2020-10-08-15_44_50.png')

news_title = news_to_name['2']

imshow(image, plot_1, title=news_title);

# Convert from the class integer encoding to actual flower names
news_names = [news_to_name[i] for i in classes]

# Plot the probabilities for the top 5 classes as a bar graph
plt.subplot(2,1,2)

sb.barplot(x=probs, y=news_names, color=sb.color_palette()[0]);

plt.show()


# In[ ]:
'''



