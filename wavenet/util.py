import numpy as np

def quantize_waveform(waveform, quantiles=256):
	"""
	Used to quantize the waveform to 256 (default) values.
	Returns the quantized values as well as indices.

	Data Type hardcoded to int16 for now.
	"""

	bits = 16 # hardcoded number of bits

	# normalize for [-1 to 1)
	waveform = waveform/2**(bits-1)

	# use the non-linear mmapping in paper
	waveform = np.sign(waveform)*(np.log(1+(quantiles-1)*abs(waveform))/np.log(quantiles)) # range: [-1 to 1)

	# get the indices
	indices = (1+waveform)/2 # range: [0 to 1)
	indices = (indices*256).astype('int') # range: [0 to 255]

	return waveform, indices

def denormalize_waveform(waveform):
	"""
	Used to denormalize a waveform from [-1 to 1] to int16 values.
	The data can be saved without denormalizing as well, though.
	"""

	bits = 16 # hardcoded number of bits
	waveform = (waveform*(2**(bits-1))).astype('int')

	return waveform
