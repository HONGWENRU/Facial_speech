import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



nc = 3
ndf = 64
ngf = 64

class FaceEncoder(nn.Module):
	def __init__(self, ngpu=0):
		super(FaceEncoder, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is (nc) x 128 x 128
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 64x 64
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 32 x 32
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 16 x 16
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 8 x 8
			nn.Conv2d(ndf * 8, ndf, 3, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True)
			# state size. ndf * 4 * 4
			)

		self.fc1 = nn.Linear(ndf*4*4, 1024)
		self.fc2 = nn.Linear(1024, 256)

		self.classifier_people = nn.Sequential(
			nn.Linear(256, 20)
			)
		self.classifier_emotion = nn.Sequential(
			nn.Linear(256, 8)
			)

	def forward(self, input):
		output =  self.main(input).view(input.size(0), ndf*4*4)
		return output

	def forward_class(self, input):
		fc = F.relu(self.fc2(F.relu(self.fc1( self.main(input).view(input.size(0), ndf*4*4) ))))
		people = self.classifier_people(fc)
		emotion = self.classifier_emotion(fc)
		return people, emotion

class FaceDecoder(nn.Module):
	"""docstring for FaceDecoder"""
	def __init__(self):
		super(FaceDecoder, self).__init__()
		self.main = nn.Sequential(
			nn.ConvTranspose2d(	 ndf*4*4, ngf * 32, 4, 1, 0, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
			nn.ReLU(True),
		)

	def forward(self, input):
		return self.main(input)

class FaceFeatureGenerator(nn.Module):
	"""docstring for FaceFeatureGenerator"""
	def __init__(self):
		super(FaceFeatureGenerator, self).__init__()
		self.main = nn.Sequential(
			nn.ConvTranspose2d(	 ndf*4*4, ngf * 32, 4, 1, 0, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
			nn.ReLU(True),
		)

ncs=1
nds=5
ngs=5

class SpeechEncoder(nn.Module):
    def __init__(self, ngpu=0):
        super(SpeechEncoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is nc x 4800
			nn.Conv1d(ncs, nds, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 2400
			nn.Conv1d(nds, nds * 2, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 1200
			nn.Conv1d(nds * 2, nds * 4, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 600
			nn.Conv1d(nds * 4, nds * 8, 4, 2, 1, bias=False),
			#nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 300
			nn.Conv1d(nds * 8, nds, 4, 3, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True)
			# state size. ndf * 100
            )

        self.fc1 = nn.Linear(nds*100, 256)

        self.classifier_people = nn.Sequential(
			nn.Linear(256, 20)
			)
        self.classifier_emotion = nn.Sequential(
			nn.Linear(256, 8)
			)
    def forward(self, input):
        output =  self.main(input).view(input.size(0), nds*100)
        return output

    def forward_class(self, input):
        sc = F.relu(self.fc1( self.main(input).view(input.size(0), nds*100) ))
        people = self.classifier_people(sc)
        emotion = self.classifier_emotion(sc)
        return people, emotion

class SpeechDecoder(nn.Module):
    def __init__(self):
		super(SpeechDecoder, self).__init__()
		self.main = nn.Sequential(
			# 1 x 1000
			nn.ConvTranspose1d(	 nds*100, ngs * 400, 7, 3, 0, bias=False),
			nn.ReLU(True),
			# 1 x 4000 x 8
			nn.ConvTranspose1d(ngs * 400, ngs * 200, 7, 5, 0, bias=False),
			nn.ReLU(True),
			# 1 x 2000 x 11
			nn.ConvTranspose1d(ngs * 200, ngs * 50, 11, 5, 0, bias=False),
			nn.ReLU(True),
			# 1 x 500 x 44
			nn.ConvTranspose1d(ngs * 50, ngs*10, 9, 5, 0, bias=False),
			nn.ReLU(True),
			nn.ConvTranspose1d(ngs * 10, ncs, 10, 5, 0, bias=False),
			#nn.Tanh()
			# 1 x 4800
		)
    def forward(self, input):
        return self.main(input)

class SpeechFeatureGenerator(nn.Module):
    def __init__(self):
		super(SpeechFeatureGenerator, self).__init__()
		self.main = nn.Sequential(
			nn.ConvTranspose1d(	 nds*100, ngs * 600, 4, 1, 0, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose1d(ngs * 600, ngs * 200, 4, 3, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose1d(ngs * 200, ngs * 100, 4, 2, 1, bias=False),
			nn.ReLU(True),

			nn.ConvTranspose1d(ngs * 100, ngs * 50, 4, 2, 1, bias=False),
			nn.ReLU(True),

            nn.ConvTranspose1d(ngs * 50, ngs * 25, 4, 2, 1, bias=False),
			nn.ReLU(True),

            nn.ConvTranspose1d(ngs * 25, ncs, 4, 2, 1, bias=False),
			nn.ReLU(True),
		)


'''
def out(i, k, s):
    return (i-1)*s + k

out(out(out(out(out(1,6,3), 6, 5),10,5),8,5),10,5)

(((((((((x-1)*s +k)-1)*s+k)-1)*s+k)-1)*s+k)-1)*s+k = 4800

10, 5 /  959
9,  5 /  191
11, 5 /  37
7,  5 /  7

def inp(o, k, s):
	return (o-k)/s + 1

inp(inp(inp(inp(inp(4800,10,5), 8, 5),10,5),6,5),6,3)
'''