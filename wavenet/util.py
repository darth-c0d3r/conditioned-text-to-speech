import numpy as np
import torch
import os
import scipy.io.wavfile as wavfile

class Hyperparameters():
	"""
	Empty class that can hold the hyperparameters
	"""
	def __init__(self):
		pass

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
	Indices : [0,quantiles-1]
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
	indices = (indices*quantiles).astype('int') # range: [0 to 255]

	# get the waveform from the indices
	waveform = 2*(indices.astype('float')/quantiles) - 1

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

def index2normalize(indices, quantiles=256):
	"""
	converts a sequence of indices to corresponding values in [-1 to 1)
	"""
	waveform = float(indices)/float(quantiles) # [0 to 1)
	waveform = (waveform*2)-1 # [-1 to 1)

	return waveform

def save_model(model, filename=None):
	"""
	Used to save the model into a file.
	"""
	folder = "saved_models"
	if folder not in os.listdir():
		os.mkdir(folder)
	folder = folder+"/"

	if filename is None:
		while True:
			files = os.listdir(folder)
			filename = input("Enter filename [model]: ")
			if filename in files:
				response = input("Warning! File already exists. Override? [y/n] : ")
				if response.strip() in ("Y", "y"):
					break
				continue
			break

	torch.save(model, folder+filename)

def save_audio(data, rate, filename=None):
	"""
	Used to save the audio into a file.
	"""
	folder = "audio_samples"
	if folder not in os.listdir():
		os.mkdir(folder)
	folder = folder+"/"

	if filename is None:
		while True:
			files = os.listdir(folder)
			filename = input("Enter filename [audio]: ")
			if filename in files:
				response = input("Warning! File already exists. Override? [y/n] : ")
				if response.strip() in ("Y", "y"):
					break
				continue
			break

	wavfile.write(folder+filename, rate, data)