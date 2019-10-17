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

		self.quantiles = quantiles # number of quantizations of data
		self.sample_rate = None # will be set externally after training

		# the first layer is the causal layer to remove temporal dependencies
		self.causal_conv = nn.Conv1d(1, res_channels, kernel_size, padding=kernel_size)

		# initialize the empty list
		self.res_layers = nn.ModuleList()

		for layer in range(num_layers):

			# calculate dilation and padding
			dilation = 2**(layer%max_dilation)
			padding = kernel_size * dilation - (dilation - 1) - 1

			# 2 conv layers for gated activation unit
			conv1 = nn.Conv1d(res_channels, res_channels, kernel_size, dilation=dilation, padding=padding)
			conv2 = nn.Conv1d(res_channels, res_channels, kernel_size, dilation=dilation, padding=padding)

			# 1x1 conv layers for residual layers and skip connections
			res_conv = nn.Conv1d(res_channels, res_channels, 1)
			skip_conv = nn.Conv1d(res_channels, skip_channels, 1)

			self.res_layers.append(nn.ModuleList([conv1, conv2, res_conv, skip_conv]))
			
		# output layer will have a categorical loss
		self.output_layer1 = nn.Conv1d(num_layers*skip_channels, output_size, 1)
		self.output_layer2 = nn.Conv1d(output_size, output_size, 1)

	def forward(self, X):

		# X.shape = [k,1,n]
		# where k is batch size

		inp_len = X.shape[2] # get the input seq length

		X = self.causal_conv(X)[:,:,:inp_len] # remove the temporal dependencies
		skip_outs = []

		for conv1, conv2, res_conv, skip_conv in self.res_layers:

			# gated activation unit + residual connection
			dilation_out = (torch.tanh(conv1(X))*torch.sigmoid(conv2(X)))[:,:,:inp_len]
			X = X + res_conv(dilation_out)

			# add the skip connection output
			skip_outs.append(skip_conv(dilation_out))

		# concatenate the skip-connection outputs 
		# the alternative might be to add them
		skip_outs = torch.cat(skip_outs,1)

		skip_outs = torch.relu(skip_outs)
		skip_outs = torch.relu(self.output_layer1(skip_outs))
		skip_outs = self.output_layer2(skip_outs)

		# get the final output
		return skip_outs


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