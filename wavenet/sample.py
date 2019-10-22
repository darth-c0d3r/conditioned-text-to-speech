import torch
import numpy as np
import scipy.io.wavfile as wavfile
from util import *

def sample(model, output_len, device, init=None):
	"""
	Used to sample from the learned distribution using the input model.
	"""
	print("Sampling Audio...")
	model.eval()

	# initialize the waveform as empty
	waveform = torch.zeros((1,model.quantiles,0)).to(device)
	indices = []

	# initialize the waveform as an input audio file
	if init is not None:
		_, waveform = wavfile.read(init)
		_, waveform = quantize_waveform(waveform, model.quantiles)
		indices = [float(idx) for idx in list(waveform)]
		waveform = torch.tensor(index2oneHot(waveform, model.quantiles))
		waveform = waveform.float().view(1,model.quantiles,-1).to(device)

	# iterate for output_len number of steps for sequential generation
	with torch.no_grad():
		for _ in range(output_len):

			probs = model(waveform)[0,:,-1]

			# use multinomial or argmax
			# idx = torch.multinomial(torch.softmax(probs,0), 1)
			idx = torch.argmax(probs)

			indices.append(float(idx))

			onehot = torch.zeros((1,model.quantiles,1))
			onehot[0,idx,0] = 1
			onehot = onehot.to(device)

			waveform = torch.cat([waveform, onehot], 2)

	return index2normalize(np.array(indices), model.quantiles)

if __name__ == "__main__":

	folder = "saved_models/"
	modelname = "wavenet1.pt"
	device = get_device()

	length = 20000

	model = torch.load(folder+modelname, map_location=device).to(device)
	audio_sample = sample(model, length, device)
	save_audio(audio_sample, model.sample_rate, "sample8.wav")
