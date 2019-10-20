import torch
import torch.nn as nn
from util import *

class Wavenet(nn.Module):
	"""
	Class for a generic Unconditioned Wavenet Model.

	All the layers in the network are Residual Layers with Skip Connections.
	Dilation will increase exponentially by a factor of 2.

	"""

	def __init__(self, quantiles=256):

		super(Wavenet, self).__init__()

		# --------------------------------#
		# network parameters
		num_layers = 8 # number of residual layers in the network
		max_dilation = 4 # after how many layers to reset dilation
		kernel_size = 3 # kernel size of the layers (excluding dilation)
		res_channels = 32 # number of channels in the residual connections
		skip_channels = quantiles # number of channels in the skip connections
		output_size = quantiles # output size : equal to the number of quantizations

		# --------------------------------#

		self.skip_channels = skip_channels # needed to init skip_sum later
		self.res_channels = res_channels # needed to split for gated conv
		self.quantiles = quantiles # number of quantizations of data
		self.sample_rate = None # will be set externally after training
		self.device = None # Required for the inference hack

		# the first layer is the causal layer to remove temporal dependencies
		self.causal_conv = nn.Conv1d(quantiles, res_channels, kernel_size, padding=kernel_size)

		# initialize the empty list
		self.res_layers = nn.ModuleList()

		for layer in range(num_layers):

			# calculate dilation and padding
			dilation = 2**(layer%max_dilation)
			padding = kernel_size * dilation - (dilation - 1) - 1

			# joined conv layer for gated activation unit
			gated_conv = nn.Conv1d(res_channels, 2*res_channels, kernel_size, dilation=dilation, padding=padding)

			# 1x1 conv layers for residual layers and skip connections
			res_conv = nn.Conv1d(res_channels, res_channels, 1)
			skip_conv = nn.Conv1d(res_channels, skip_channels, 1)

			self.res_layers.append(nn.ModuleList([gated_conv, res_conv, skip_conv]))
			
		# output layer will have a categorical loss
		# self.output_layer1 = nn.Conv1d(num_layers*skip_channels, output_size, 1)
		# Uncomment above if appending skip connection outputs
		self.output_layer1 = nn.Conv1d(skip_channels, output_size, 1)
		self.output_layer2 = nn.Conv1d(output_size, output_size, 1)

	def forward(self, X):

		# X.shape = [k,1,n]
		# where k is batch size

		# a little hack for first step of inference
		if X.shape[2] == 0:
			X = torch.zeros((X.shape[0], X.shape[1], 1)).to(self.device)

		inp_len = X.shape[2] # get the input seq length

		X = torch.relu(self.causal_conv(X))[:,:,:inp_len] # remove the temporal dependencies
		skip_outs = []
		skip_sum = torch.zeros(X.shape[0], self.skip_channels, X.shape[2])

		for gated_conv, res_conv, skip_conv in self.res_layers:

			# gated activation unit + residual connection
			gated_out = gated_conv(X)

			# split the joint output into two
			tanh_in = gated_out[:,:self.res_channels,:]
			sigm_in = gated_out[:,self.res_channels:,:]

			# get the dilation output
			dilation_out = (torch.tanh(tanh_in)*torch.sigmoid(sigm_in))[:,:,:inp_len]

			# add the residual layer output to input
			X = X + res_conv(dilation_out)

			# add the skip connection output
			skip_out = skip_conv(dilation_out)
			skip_outs.append(skip_out)
			skip_sum += skip_out

		# concatenate the skip-connection outputs 
		# the alternative might be to add them
		# Uncomment the one that you want to use
		# Remember to change the layer size in __init__ accordingly

		# skip_outs = torch.cat(skip_outs,1)
		skip_outs = skip_sum

		skip_outs = torch.relu(skip_outs)
		skip_outs = torch.relu(self.output_layer1(skip_outs))
		skip_outs = self.output_layer2(skip_outs)

		# get the final output
		return skip_outs

	def weight_init(self, params):
		"""
		Used to initialize the weights of all the layers.
		params contain params.type and other values needed for init.
		params.type can be one of the following:
		1. uniform (params.a, params.b needed)
		2. normal (params.mean, params.std needed)
		3. xavier-uniform
		4. xavier-normal
		Need to complete it later.
		"""
		pass



class Convnet(nn.Module):
	"""
	Class for a simple 1D Convnet without Residual Connections.
	Set max_dilation = 1 to turn off dilation
	"""

	def __init__(self, quantiles=256):

		super(Convnet, self).__init__()

		# --------------------------------#
		# network parameters
		
		num_layers = 3 # number of layers in the network
		max_dilation = 1 # after how many layers to reset dilation
		kernel_size = 5
		output_size = quantiles

		# --------------------------------#

		# initialize the empty list
		self.conv_layers = nn.ModuleList()

		for layer in range(num_layers):

			# calculate dilation and padding
			dilation = 2**(layer%max_dilation)
			padding = kernel_size * dilation - (dilation - 1)

			self.conv_layers.append(nn.Conv1d(1,1,kernel_size,dilation=dilation,padding=padding))
			
		# output layer will have a categorical loss
		self.output_layer = nn.Conv1d(1,output_size,1)

	def forward(self, X):

		# X.shape = [1,1,n]
		# try [1,k,n] later maybe

		inp_len = X.shape[2]

		for conv in self.conv_layers:

			X = torch.relu(conv(X))[:,:,:inp_len]

		X = self.output_layer(X)
		return X