import torch
import torch.nn as nn
import torchvision
import time
import os
from PIL import Image
from torchsummary import summary

class OZLoss(nn.Module):
	def __init__(self, loss):
		super(OZLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(1.0))
		self.register_buffer('fake_label', torch.tensor(0.0))
		self.loss = loss

	def get_target_tensor(self, prediction, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(prediction).to(torch.device('cuda'))

	def __call__(self, prediction, target_is_real):
		target_tensor = self.get_target_tensor(prediction, target_is_real)
		loss = self.loss(prediction, target_tensor)
		return loss

class ImageData(torch.utils.data.Dataset):

	def __init__(self,root_dir,transform=None):
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(os.listdir(self.root_dir))

	def __getitem__(self, idx):
		img_name = os.listdir(self.root_dir)[idx]
		image = Image.open(self.root_dir+img_name)
		image = image.convert("RGB")
		return self.transform(image)

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		#Input
		layers = [nn.ReflectionPad2d(padding=3),
				 nn.Conv2d(in_channels=3,
						   out_channels=64,
						   kernel_size=7,
						   padding=0,
						   bias=True),
				 nn.InstanceNorm2d(num_features=64),
				 nn.ReLU(inplace=True)]

		#Downsampling
		layers += [nn.Conv2d(in_channels=64	,
							out_channels=128,
							kernel_size=3,
							stride=2,
							padding=1,
							bias=True),
				  nn.InstanceNorm2d(num_features=128),
				  nn.ReLU(inplace=True)]

		layers += [nn.Conv2d(in_channels=128,
					out_channels=256,
					kernel_size=3,
					stride=2,
					padding=1,
					bias=True),
		  nn.InstanceNorm2d(num_features=256),
		  nn.ReLU(inplace=True)]

		for i in range(6):
			layers += [ResNetBlock()]

		#Upsampling
		layers += [nn.ConvTranspose2d(in_channels=256,
									 out_channels=128,
									 kernel_size=3,
									 stride=2,
									 padding=1,
									 output_padding=1,
									 bias=True),
				  nn.InstanceNorm2d(num_features=128),
				  nn.ReLU(inplace=True)]

		layers += [nn.ConvTranspose2d(in_channels=128,
									 out_channels=64,
									 kernel_size=3,
									 stride=2,
									 padding=1,
									 output_padding=1,
									 bias=True),
				  nn.InstanceNorm2d(num_features=64),
				  nn.ReLU(inplace=True)]

		layers += [nn.ReflectionPad2d(3)]
		layers += [nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)]
		layers += [nn.Tanh()]

		self.model = torch.nn.Sequential(*layers)

	def forward(self,input):
		return self.model(input)

class ResNetBlock(nn.Module):
	def __init__(self):

		super(ResNetBlock, self).__init__()
		resnet_block = [nn.ReflectionPad2d(padding=1)]
		resnet_block += [nn.Conv2d(in_channels=256,
								 out_channels=256,
								 kernel_size=3,
								 padding=0,
								 bias=True),
					   nn.InstanceNorm2d(num_features=256),
					   nn.ReLU(inplace=True)]

		resnet_block += [nn.ReflectionPad2d(padding=1)]
		resnet_block += [nn.Conv2d(in_channels=256,
								 out_channels=256,
								 kernel_size=3,
								 padding=0,
								 bias=True),
					   nn.InstanceNorm2d(num_features=256)]

		self.model = nn.Sequential(*resnet_block)

	def forward(self,input):
		return input+self.model(input)

class Discriminator(nn.Module):
	def __init__(self):

		super(Discriminator, self).__init__()
		layers = [nn.Conv2d(in_channels=3,
							  out_channels=64,
							  kernel_size=4,
							  stride=2,
							  padding=1),
					nn.LeakyReLU(negative_slope=0.2,
								 inplace=True)]

		layers += [nn.Conv2d(in_channels=64,
							   out_channels=128,
							   kernel_size=4,
							   stride=2,
							   padding=1,
							   bias=True),
					 nn.InstanceNorm2d(num_features=128),
					 nn.LeakyReLU(0.2, True)]

		layers += [nn.Conv2d(in_channels=128,
							   out_channels=256,
							   kernel_size=4,
							   stride=2,
							   padding=1,
							   bias=True),
					 nn.InstanceNorm2d(num_features=256),
					 nn.LeakyReLU(0.2, True)]

		layers += [nn.Conv2d(in_channels=256,
							   out_channels=512,
							   kernel_size=4,
							   stride=1,
							   padding=1,
							   bias=True),
					 nn.InstanceNorm2d(num_features=512),
					 nn.LeakyReLU(negative_slope=0.2,
								  inplace=True)]

		layers += [nn.Conv2d(in_channels=512,
							   out_channels=1,
							   kernel_size=4,
							   stride=1,
							   padding=1)]

		self.model = nn.Sequential(*layers)

	def forward(self,input):
		return self.model(input)
