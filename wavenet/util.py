import numpy as np
import torch

def get_device(cuda=True):
	"""
	returns the device used in the system (cpu/cuda)
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device : " + str(device))
	return device

def quantize_waveform(waveform, quantiles=256, non_linear=True):
	"""
	Used to quantize the waveform to 256 (default) values.
	Returns the quantized values as well as indices.

	Data Type hardcoded to int16 for now.

	Input waveform : [-2^x,2^x-1]
	Output waveform : [-1,1)
	Indices : [0,255]
	"""

	bits = 16 # hardcoded number of bits

	# normalize for [-1 to 1)
	waveform = waveform/2**(bits-1)

	# use the non-linear mapping from the paper
	if non_linear is True:
		waveform = np.sign(waveform)*(np.log(1+(quantiles-1)*abs(waveform))/np.log(quantiles)) # range: [-1 to 1)
	else:
		waveform = np.array(waveform)

	# get the indices
	indices = (1+waveform)/2 # range: [0 to 1)
	indices = (indices*256).astype('int') # range: [0 to 255]

	# get the waveform from the indices
	waveform = 2*(indices.astype('float')/256) - 1

	return waveform, indices

def normalize2denormalize(waveform):
	"""
	Used to denormalize a waveform from [-1 to 1) to int16 values.
	The data can be saved without denormalizing as well, though.
	So it is not required so far.
	"""

	bits = 16 # hardcoded number of bits
	waveform = (waveform*(2**(bits-1))).astype('int')

	return waveform

def index2normalize(indices):
	"""
	converts a sequence of indices to corresponding values in [-1 to 1)
	"""
	waveform = float(indices)/256.0 # [0 to 1)
	waveform = (waveform*2)-1 # [-1 to 1)

	return waveform

def sample(model, output_len, device):
	"""
	Used to sample from the learned distribution using the input model.
	"""

	waveform = [0.]
	for _ in range(output_len):
		data = torch.tensor(waveform).view(1,1,-1).to(device)
		probs = model(data)[0,:,-1]
		idx = torch.multinomial(probs, 1)
		waveform.append(index2normalize(idx))

	return np.array(waveform[1:])