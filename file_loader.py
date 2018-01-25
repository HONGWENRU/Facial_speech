import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import subprocess as sp
import cv2


FFMPEG_BIN = "ffmpeg"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

emotion_dict = {1:"neutral", 2:"calm", 3:"happy", 4:"sad", 5:"angry", 6:"fearful", 7:"disgust", 8:"surprised"}

def deformate_file_name(file_name):
	# 01-01-01-01-01-01-01.mp4
	'''
	Modality (1 = Audio-Video, 2 = Video-only, 3 = Audio-only)
	Vocal channel (1 = speech, 2 = song)
	Emotion
	   Speech (1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful, 7 = disgust, 8 = surprised)
	   Song   (1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fearful)
	Emotional intensity (1 = normal, 2 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
	Statement (1 = Kids are talking by the door, 2 = Dogs are sitting by the door)
	Repetition (1 = 1st rep, 2 = 2nd rep)
	Actor (1 to 24. Odd = male, Even = female)
	'''
	f_name = os.path.basename(file_name)
	numbers = f_name.split('-')
	if numbers[0] != '01' or numbers[1] != '01':
		return "",""
	emotion = emotion_dict[int(numbers[2])]
	intensity = "normal" if numbers[3]=="01" else "strong"
	statement = "kidtalk" if numbers[4]=="01" else "dogsit"
	repetition = numbers[5]
	out_name = "%s_%s_%s_actor_%s" % (intensity, statement, repetition, numbers[6][:-4])
	return out_name, emotion


def get_frame_from_video(video_path, devided_by=30, out_path='train/'):
	command = [ FFMPEG_BIN,
             '-i', video_path,
             '-f', 'image2pipe',
             '-pix_fmt', 'rgb24',
             '-vcodec', 'rawvideo', '-']
	pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
	total_buff = pipe.stdout.read()
	total_images = np.fromstring(total_buff, dtype='uint8')
	total_image_number = total_images.shape[0] / VIDEO_HEIGHT / VIDEO_WIDTH / 3
	step = total_image_number / devided_by
	diff = total_image_number % devided_by
	out_name, emotion = deformate_file_name(video_path)
	if out_name=="":
		return
	if not os.path.exists(os.path.join(out_path, emotion)):
		os.mkdir(os.path.join(out_path, emotion))
	for i in range(devided_by):
		image = total_images[(i*step+1)*VIDEO_WIDTH*VIDEO_HEIGHT*3: (i*step+2)*VIDEO_HEIGHT*VIDEO_WIDTH*3]
		image = image.reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3))
		image = image[:,(VIDEO_WIDTH - VIDEO_HEIGHT)/2 : (VIDEO_WIDTH - VIDEO_HEIGHT)/2 + VIDEO_HEIGHT,(2,1,0)]
		image = cv2.resize(image, (128, 128))
		cv2.imwrite( os.path.join( out_path, emotion, '%s_clip_%d.jpg'%( out_name, i+1)) , image)
	print total_image_number
	pipe.stdout.flush()


def get_audio_from_video(video_path, divided_by=30, sample_length=4800, out_path='train/'):
	command = [ FFMPEG_BIN, "-i", video_path, "-f", "wav", "-"]
	p = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
	total_buff = p.communicate()[0]
	total_audio = np.fromstring(total_buff[total_buff.find("data")+4:], np.int16)
	length = total_audio.shape[0]
	diff = ( length - divided_by * sample_length )/2
	out_name, emotion = deformate_file_name(video_path)
	if out_name=="":
		return
	if not os.path.exists(os.path.join(out_path, emotion)):
		os.mkdir(os.path.join(out_path, emotion))
	for i in range(divided_by):
		sample = total_audio[ diff + i*sample_length: diff + (i+1)*sample_length ]
		np.save( os.path.join( out_path, emotion, '%s_clip_%d.npy'%(out_name, i+1)), sample)

def convert_raw_data():
	j=0
	for i in [1,5,9,13,17]:
		root = './SpeechAV_Actors%d-%d/'%(i,i+3)
		for file_name in os.listdir(root):
			get_frame_from_video(root+file_name)
			get_audio_from_video(root+file_name)
			j += 1
			print j
	root = './SpeechAV_Actors%d-%d/'%(21,21+3)
	for file_name in os.listdir(root):
		get_frame_from_video(root+file_name, out_path='test/')
		get_audio_from_video(root+file_name, out_path='test/')



class FaceTransform(object):
	"""docstring for FaceTransform"""
	def __init__(self):
		super(FaceTransform, self).__init__()
		self.normTransfer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	def __call__(self, image):
		# input is a 128 * 128 * 3 numpy array image
		image = torch.from_numpy(image.transpose((2,0,1))).float()
		if image.max() > 1.5:
			image = image/255
		#image = self.normTransfer(image)
		return image

	def deprocess(self, image):
		# input is a torch tensor of shape batch_size * 3 * 128 * 128
		# this function is not finished yet and I find it not needed
		image = image.cpu()
		image = image.numpy()
		image = image.transpose((0, 2, 3, 1))

class FaceAEDataset(Dataset):
	"""docstring for FaceAEDataset"""
	def __init__(self, root_dir):
		super(FaceAEDataset, self).__init__()
		self.root_dir = root_dir
		self.emotion_dict = {1:"neutral", 2:"calm", 3:"happy", 4:"sad", 5:"angry", 6:"fearful", 7:"disgust", 8:"surprised"}
		img_list = []
		for emotion in self.emotion_dict.values():
			for f_name in os.listdir(os.path.join(root_dir, emotion)):
				if '.jpg' in f_name:
					img_list.append( os.path.join(root_dir, emotion, f_name) )
		self.frame = img_list
		self.transform = FaceTransform()

	def __len__(self):
		return len(self.frame)

	def __getitem__(self, idx):
		img_name = self.frame[idx]
		emotion_label =  self.emotion_dict.keys()[self.emotion_dict.values().index(img_name.split('/')[-2])]
		person_label = int(os.path.basename(img_name).split('_')[4])
		image = io.imread(img_name)
		# apply transformation on images
		if self.transform:
			image = self.transform(image)
		sample = [image, emotion_label, person_label]
		return sample
'''
class SpeechTransform(object):
	"""docstring for SpeechTransform"""
	def __init__(self):
		super(SpeechTransform, self).__init__()

	def __call__(self, speech):
		# input is a 4800*1 numpy array speech
		speech = torch.from_numpy(speech)
		# normilize speech to [-1,+1]
		speech=speech

		return speech
'''
class SpeechAEDataset(Dataset):
	"""docstring for FaceAEDataset"""
	def __init__(self, root_dir):
		super(SpeechAEDataset, self).__init__()
		self.root_dir = root_dir
		self.emotion_dict = {1:"neutral", 2:"calm", 3:"happy", 4:"sad", 5:"angry", 6:"fearful", 7:"disgust", 8:"surprised"}
		speech_list = []
		for emotion in self.emotion_dict.values():
			for f_name in os.listdir(os.path.join(root_dir, emotion)):
				if '.npy' in f_name:
					speech_list.append( os.path.join(root_dir, emotion, f_name) )
		self.frame = speech_list
		self.transform = False
	def __len__(self):
		return len(self.frame)

	def __getitem__(self, idx):
		speech_name = self.frame[idx]
		emotion_label =  self.emotion_dict.keys()[self.emotion_dict.values().index(speech_name.split('/')[-2])]
		person_label = int(os.path.basename(speech_name).split('_')[4])
		speech = np.load(speech_name)
		speech = torch.from_numpy(speech)
		speech = speech.float()
		speech = speech.view(1,4800)
		# apply transformation on speech
		if self.transform:
			speech = speech/32768.0
		sample = [speech, emotion_label, person_label]
		return sample
