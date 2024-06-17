import torch 
from PIL import Image
import os 
from einops import repeat
from torchvision import transforms
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt 
from einops import rearrange
from torch.optim import Adam
from typing import Dict, Tuple
from tqdm import tqdm
import requests

from torchvision import transforms
import numpy as np
from typing import Any, Callable, Optional, Tuple, Union
from pathlib import Path
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import Bottleneck


import torch.utils.model_zoo as model_zoo
import math
batch_size =128

def conv3x3(in_planes, out_planes, stride = 1):
	return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)


class BasicBlock(nn.Module):

	expansion = 1

	def __init__(self, inplanes, outplanes, stride = 1, downsample = None):
		super().__init__()
		self.conv1 = conv3x3(inplanes, outplanes, stride)
		self.bn1 = nn.BatchNorm2d(outplanes)
		self.relu = nn.ReLU(inplace = True)
		
		self.conv2 = conv3x3(outplanes, outplanes)
		self.bn2 = nn.BatchNorm2d(outplanes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):

		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if(self.downsample is not None):
			residual = self.downsample(x)

		out += residual

		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, outplanes, stride = 1, downsample = None):
		super().__init__()
		self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(outplanes)
		
		self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(outplanes)

		self.conv3 = nn.Conv2d(outplanes, outplanes*self.expansion, kernel_size = 1, bias = False)
		self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)

		self.relu = nn.ReLU(inplace = True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x): 
		residual = x 

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if(self.downsample is not None):
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out



class ResNet(nn.Module):

	def __init__(self, block, layer, num_class = 1000, input_chanel = 3):
		super().__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(input_chanel, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace = True)
		self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

		self.layer1 = self.make_layer(block, 64, layer[0])
		self.layer2 = self.make_layer(block, 128, layer[1], stride = 2)
		self.layer3 = self.make_layer(block, 256, layer[2], stride = 2)
		self.layer4 = self.make_layer(block, 512, layer[3], stride = 2)

		self.avgpool = nn.AvgPool2d(7, stride = 2)

		self.dropout = nn.Dropout2d(p = 0.5, inplace = True)

		self.fc = nn.Linear(512*block.expansion, num_class)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2.0/n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def make_layer(self, block, planes, blocks, stride = 1):
		downsample = None

		if(stride !=1 or self.inplanes != planes * block.expansion):
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, stride = stride, bias = False), 
				nn.BatchNorm2d(planes*block.expansion))

		layers = []

		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))


		return nn.Sequential(*layers)

	def fc_params(self):
		params = []
		for name, param in self.named_parameters():
			if 'fc' in name:
				params.append(param)
		return params

	def backbone_params(self):
		params = []
		for name, param in self.named_parameters():
			if 'fc' not in name:
				params.append(param)
		return params

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = self.dropout(x)

		x = x.view(x.size(0), -1)

		x = self.fc(x)

		return x
	


class HotDogClassifier:
    def __init__(self, checkpoint_path='./hotdogdetect_checkpoint.pth', 
				 checkpoint_url='https://www.doc.ic.ac.uk/~bkainz/teaching/DL/hotdogdetect_checkpoint.pth', device='cuda'):
        self.model = ResNet(Bottleneck, [3, 4, 6, 3], 101, 3)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_url = checkpoint_url
        
        if not os.path.exists(checkpoint_path):
            self._download_checkpoint()
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.transform1 = transforms.Compose([
            transforms.Resize((224, 224))
        ])
        
    def _download_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        response = requests.get(self.checkpoint_url)
        with open(self.checkpoint_path, 'wb') as f:
            f.write(response.content)
    
    def predict(self, images):
        if not isinstance(images, torch.Tensor):
            images = self._transform_and_to_tensor(images)
        
        images =  self.transform1(images.detach().to(self.device))
        outputs = self.model(images)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        _, top5_pred = torch.topk(outputs, 5, dim=1)
        correct_top5 = torch.sum(top5_pred == 55).item()
        total_samples = images.size(0)
        top5_accuracy = correct_top5 / total_samples
        return predictions, probabilities, top5_accuracy
    
    def _transform_and_to_tensor(self, images):
        if isinstance(images, list) or isinstance(images, Dataset):
            images = torch.stack([self.transform(image) for image in images])
        else:
            images = self.transform(images).unsqueeze(0)
        return images