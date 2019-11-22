from util import *
import librosa
from torch.utils.data import Dataset
import os

class SpeakerDataset(Dataset):

	def __init__(self, num_features, folder):

		if folder.endswith("/"):
			folder = folder[:-1]

		self.num_features = num_features
		self.folder = folder

		self.speakers = os.listdir(folder)
		self.num_speakers = len(self.speakers)
		# self.samples_per_speaker = [len(os.listdir(self.folder + "/" + speaker)) for speaker in self.speakers]
		self.samples_per_speaker = len(os.listdir(self.folder + "/" + self.speakers[0])) # assuming equal samples

		# this is temporary
		# finally, we should break each clip into subparts and calculate average of features

	def __len__(self):
		# number of possible clips
		# = (n*m)
		return self.num_speakers * self.samples_per_speaker

	def __getitem__(self, idx):
		
		spkr, file = idx // self.samples_per_speaker, idx % self.samples_per_speaker
		file = self.folder+"/"+self.speakers[spkr]+"/"+os.listdir(self.folder+"/"+self.speakers[spkr])[file]

		rate, data = librosa.load(file)
		features = librosa.features.mfcc(data, rate, n_mfcc=self.num_features)

		# visualize_waveform(waveform=data)

		return torch.tensor(features).t(), torch.tensor(spkr)

def getSpeakerDataset(folder):
	num_features = 20
	return {"data" : SpeakerDataset(num_features, folder)}

# ---------------------------------------------------------------------------------------------- #

class SpeakerPairsDataset(Dataset):

	def __init__(self, num_features, folder):

		if folder.endswith("/"):
			folder = folder[:-1]

		self.num_features = num_features
		self.folder = folder

		self.speakers = os.listdir(folder)
		self.num_speakers = len(self.speakers)

		self.voice_clips = []
		self.speaker_tags = []

		for tag in range(len(self.speakers)):
			speaker_folder = self.folder + "/" + self.speakers[tag] + "/"
			speaker_clips = os.listdir(speaker_folder)
			for clip in speaker_clips:
				self.voice_clips.append(speaker_folder+clip)
				self.speaker_tags.append(tag)

		# this is temporary
		# finally, we should break each clip into subparts and calculate average of features



	def __len__(self):
		# number of possible pairs
		# = (n*m)*(n*m)

		return (len(self.voice_clips)) ** 2

	def __getitem__(self, idx):
		
		idx1 = idx // len(self.voice_clips) # [0, n*m)
		idx2 = idx  % len(self.voice_clips) # [0, n*m)

		file1 = self.voice_clips[idx1]
		file2 = self.voice_clips[idx2]

		spkr1 = self.speaker_tags[idx1]
		spkr2 = self.speaker_tags[idx2]

		data1, rate1 = librosa.load(file1)
		data2, rate2 = librosa.load(file2)

		features1 = librosa.feature.mfcc(data1, rate1, n_mfcc=self.num_features)
		features2 = librosa.feature.mfcc(data2, rate2, n_mfcc=self.num_features)

		# visualize_waveform(waveform=data1)
		# visualize_waveform(waveform=data2)

		return torch.tensor(features1).t().float(), torch.tensor(features2).t().float(), torch.tensor(int(spkr1 == spkr2))

def getSpeakerPairsDataset(folder):
	num_features = 20
	return {"data" : SpeakerPairsDataset(num_features, folder)}


# ---------------------------------------------------------------------------------------------- #

class DummyDataset(Dataset):

	def __init__(self, quantiles, num_speakers, samples_per_speaker, sample_length):

		self.quantiles = quantiles # = q
		self.num_speakers = num_speakers # = n
		self.samples_per_speaker = samples_per_speaker # = m
		self.sample_length = sample_length # = l

		self.bits = 16

	def __len__(self):
		# number of possible pairs
		# = (n*m)*(n*m)

		return (self.num_speakers * self.samples_per_speaker) ** 2

	def __getitem__(self, idx):
		
		spkr1 = (idx//(self.num_speakers * self.samples_per_speaker))//self.samples_per_speaker
		spkr2 = (idx %(self.num_speakers * self.samples_per_speaker))//self.samples_per_speaker

		value1 = (float(2*spkr1)/float(self.num_speakers-1)) - 1.0
		value2 = (float(2*spkr2)/float(self.num_speakers-1)) - 1.0

		audio1 = 0.001*np.random.randn((self.sample_length)) + value1
		audio2 = 0.001*np.random.randn((self.sample_length)) + value2

		audio1 = np.clip(audio1, -1.0, 0.9999)
		audio2 = np.clip(audio2, -1.0, 0.9999)

		wvfrm1, audio1 = quantize_waveform(audio1 * (2**(self.bits-1)), self.quantiles, non_linear=False)
		wvfrm2, audio2 = quantize_waveform(audio2 * (2**(self.bits-1)), self.quantiles, non_linear=False)

		# visualize_waveform(waveform=wvfrm1)
		# visualize_waveform(waveform=wvfrm2)

		audio1 = index2oneHot(audio1, self.quantiles)
		audio2 = index2oneHot(audio2, self.quantiles)

		audio1 = torch.zeros(audio1.shape) + value1
		audio2 = torch.zeros(audio2.shape) + value2

		return torch.tensor(audio1).t().float(), torch.tensor(audio2).t().float(), torch.tensor(int(spkr1 == spkr2))

def getDummyDataset():

	quantiles = 256
	num_speakers = 4
	samples_per_speaker = 2
	sample_length = 10

	return {"data" : DummyDataset(quantiles, num_speakers, samples_per_speaker, sample_length)}
