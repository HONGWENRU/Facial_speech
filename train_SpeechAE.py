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
from file_loader import SpeechAEDataset
import time
from models import SpeechEncoder, SpeechDecoder, weights_init
import numpy as np

import matplotlib.pyplot as plt


sd=SpeechAEDataset(root_dir='/Users/hongwenfan/Desktop/Facial_Speech_Project/train')
dataloader = DataLoader(sd, batch_size=72, shuffle=True, num_workers=0)


encoder = SpeechEncoder()
decoder = SpeechDecoder()
#encoder.load_state_dict(torch.load('speechAE_encoder_10.pt'))
# decoder.load_state_dict(torch.load('speechAE_decoder_10.pt'))

criterionE = nn.CrossEntropyLoss()
criterionD = nn.MSELoss()

nepoch = 5000
cuda = True

# if cuda:
# 	encoder = encoder.cuda()
# 	decoder = decoder.cuda()
# 	criterionE = criterionE.cuda()
# 	criterionD = criterionD.cuda()

optimizerE = torch.optim.SGD(encoder.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
optimizerD = torch.optim.Adam( decoder.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerE2 = torch.optim.Adam( encoder.parameters(), lr=0.0001, betas=(0.5, 0.999))

#encoder.apply(weights_init)
#decoder.apply(weights_init)

#encoder.load_state_dict(torch.load('speechAE_encoder_10.pt'))
#decoder.load_state_dict(torch.load('speechAE_decoder_10.pt'))

def train():
	for epoch in range(0, nepoch):

		accu_loss_person = 0.0
		accu_loss_emotion = 0.0
		accu_loss_decoder = 0.0
		timer = time.time()

		for i, batch_sample in enumerate(dataloader):
			cur_batch_size = batch_sample[0].size(0)

			encoder.zero_grad()
			speech, emotion, person = batch_sample

			speech_v = Variable(speech)
			emotion_v = Variable(emotion-1)
			person_v = Variable(person-1)

			# if cuda:
			# 	speech_v = speech_v.cuda()
			# 	emotion_v = emotion_v.cuda()
			# 	person_v = person_v.cuda()

			# Train Encoder classification part
			pred_person, pred_emotion = encoder.forward_class(speech_v)

			loss_person = criterionE(pred_person, person_v)
			#print pred_person
			#print person_v
			loss_emotion = criterionE(pred_emotion, emotion_v)

			loss = loss_emotion + loss_person
			loss.backward()
			optimizerE.step()

			encoder.zero_grad()

			accu_loss_person += loss_person.data.mean()
			accu_loss_emotion += loss_emotion.data.mean()
			# Train Decoder and Encoder image generation part
			decoder.zero_grad()

			features = encoder(speech_v)
			#print features
			g_speech = decoder( features.view(-1, 500, 1) )

			loss_decoder = criterionD(g_speech, speech_v)
			loss_decoder.backward()

			accu_loss_decoder += loss_decoder.data.mean()

			optimizerD.step()
			optimizerE2.step()

			interval = 100.0
			if i%interval==0:
				print '[%d/%d][%d/%d]  loss_person: %.5f   loss_emotion: %.5f   loss_decoder: %.5f   time: %.5f' % (i, len(dataloader), epoch, nepoch, accu_loss_person/interval, accu_loss_emotion/interval, accu_loss_decoder/interval, time.time()-timer)
				accu_loss_person = 0.0
				accu_loss_emotion = 0.0
				accu_loss_decoder = 0.0
				timer = time.time()
				#utils.save_image(g_img.data, 'decoder_img_%d.jpg'%(epoch+1), normalize=False)
				#utils.save_image(img_v.data, 'decoder_img_%d_real.jpg'%(epoch+1), normalize=False)
		if (epoch+1)%50==0:
			#np.save(g_speech.data.numpy(), 'decoder_speech_%d.npy'%(epoch+1))
			#np.save(speech_v.data.numpy(), 'decoder_speech_%d_real.npy'%(epoch+1))
            #utils.save_image(g_speech.data, 'decoder_speech_%d.jpg'%(epoch+1), normalize=False)
			#utils.save_image(speech_v.data, 'decoder_speech_%d_real.jpg'%(epoch+1), normalize=False)
			try:
				torch.save(encoder.state_dict(), 'speechAE_encoder_%d.pt'%(epoch+1))
				torch.save(decoder.state_dict(), 'speechAE_decoder_%d.pt'%(epoch+1))
			except:
				pass
train()
