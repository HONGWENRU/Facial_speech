import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from file_loader import FaceAEDataset
import time
from models import FaceEncoder, FaceDecoder


fd = FaceAEDataset(root_dir='train')
dataloader = DataLoader(fd, batch_size=72, shuffle=True, num_workers=4)


encoder = FaceEncoder()
decoder = FaceDecoder()
encoder.load_state_dict(torch.load('faceAE_encoder_100.pt'))
decoder.load_state_dict(torch.load('faceAE_decoder_100.pt'))

criterionE = nn.CrossEntropyLoss()
criterionD = nn.MSELoss()

nepoch = 100
cuda = False

if cuda:
	encoder = encoder.cuda()
	decoder = decoder.cuda()
	criterionE = criterionE.cuda()
	criterionD = criterionD.cuda()

optimizerE = torch.optim.SGD(encoder.parameters(), lr=0.00002, momentum=0.9, weight_decay=1e-4)
optimizerD = torch.optim.Adam(decoder.parameters(), lr=0.00002, betas=(0.5, 0.999))

def generate():
	fd = FaceAEDataset(root_dir='test')
	dataloader = DataLoader(fd, batch_size=36, shuffle=False, num_workers=4)
	for i, batch_sample in enumerate(dataloader):
		
		cur_batch_size = batch_sample[0].size(0)

		encoder.zero_grad()
		img = batch_sample[0]
			
		img_v = Variable(img.cuda())

		features = encoder(img_v)
		g_img = decoder( features.view(cur_batch_size, -1, 1, 1) )
		utils.save_image(g_img.data, 'decoder_img_%d.jpg'%(i), normalize=False)

#generate()


def train():
	for epoch in range(0, nepoch):

		accu_loss_person = 0.0
		accu_loss_emotion = 0.0
		accu_loss_decoder = 0.0
		timer = time.time()

		for i, batch_sample in enumerate(dataloader):
			cur_batch_size = batch_sample[0].size(0)

			encoder.zero_grad()
			img, emotion, person = batch_sample
			
			img_v = Variable(img)
			emotion_v = Variable(emotion-1)
			person_v = Variable(person-1)

			if cuda:
				img_v = img_v.cuda()
				emotion_v = emotion_v.cuda()
				person_v = person_v.cuda()

			# Train Encoder classification part
			pred_person, pred_emotion = encoder.forward_class(img_v)

			loss_person = criterionE(pred_person, person_v)
			loss_emotion = criterionE(pred_emotion, emotion_v)

			loss = loss_emotion + loss_person
			loss.backward()
			optimizerE.step()

			encoder.zero_grad()

			accu_loss_person += loss_person.data[0]
			accu_loss_emotion += loss_emotion.data[0]
			# Train Decoder and Encoder image generation part
			decoder.zero_grad()
			
			features = encoder(img_v)
			#print features
			g_img = decoder( features.view(cur_batch_size, -1, 1, 1) )
			
			loss_decoder = criterionD(g_img, img_v)
			loss_decoder.backward()
			
			accu_loss_decoder += loss_decoder.data[0]

			optimizerD.step()
			optimizerE.step()

			interval = 100.0
			if i%interval==0:
				interval = interval/100
				print '[%d/%d][%d/%d]  loss_person: %.5f   loss_emotion: %.5f   loss_decoder: %.5f   time: %.5f' % (i, len(dataloader), epoch, nepoch, accu_loss_person/interval, accu_loss_emotion/interval, accu_loss_decoder/interval, time.time()-timer)
				accu_loss_person = 0.0
				accu_loss_emotion = 0.0
				accu_loss_decoder = 0.0
				timer = time.time()
				#utils.save_image(g_img.data, 'decoder_img_%d.jpg'%(epoch+1), normalize=False)
				#utils.save_image(img_v.data, 'decoder_img_%d_real.jpg'%(epoch+1), normalize=False)
		if (epoch+1)%5==0:
			utils.save_image(g_img.data, 'decoder_img_%d.jpg'%(epoch+1), normalize=False)
			utils.save_image(img_v.data, 'decoder_img_%d_real.jpg'%(epoch+1), normalize=False)
			torch.save(encoder.state_dict(), 'faceAE_encoder_%d.pt'%(epoch+1))
			torch.save(decoder.state_dict(), 'faceAE_decoder_%d.pt'%(epoch+1))













#utils.save_image(a, 'test.jpg', normalize=True)