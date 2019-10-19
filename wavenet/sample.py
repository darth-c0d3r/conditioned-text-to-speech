import torch
import numpy as np
from util import *

def sample(model, output_len, device):
	"""
	Used to sample from the learned distribution using the input model.
	"""
	print("Sampling Audio...")
	model.eval()

	# initialize waveform as empty
	waveform = torch.zeros((1,model.quantiles,0)).to(device)
	indices = []

	for _ in range(output_len):

		probs = torch.softmax(model(waveform)[0,:,-1],0)

		idx = torch.multinomial(probs, 1) # might consider argmax also
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

	length = 10000

	model = torch.load(folder+modelname).to(device)
	audio_sample = sample(model, length, device)
	save_audio(audio_sample, model.sample_rate, "sample2.wav")
