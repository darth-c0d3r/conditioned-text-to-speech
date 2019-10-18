import torch
from util import *

def sample(model, output_len, device):
	"""
	Used to sample from the learned distribution using the input model.
	"""
	print("Sampling Audio...")
	model.eval()
	waveform = [(2*(torch.rand(1).item()))-1] #  random init [-1,1]
	for _ in range(output_len):
		data = torch.tensor(waveform).view(1,1,-1).to(device)
		probs = torch.softmax(model(data),2)[0,:,-1]
		idx = torch.multinomial(probs, 1)
		waveform.append(index2normalize(idx, model.quantiles))

	return np.array(waveform[1:])

if __name__ == "__main__":

	folder = "saved_models/"
	modelname = "wavenet1.pt"
	device = get_device()

	length = 10000

	model = torch.load(folder+modelname).to(device)
	audio_sample = sample(model, length, device)
	save_audio(audio_sample, model.sample_rate, "sample2.wav")
