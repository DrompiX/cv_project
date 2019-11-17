import torch
import torch.nn as nn
import torchvision
import time
import os
from PIL import Image
from torchsummary import summary
from model import *

def trainOnFruitPair(fruitNameA,fruitNameB,epochs):
	if (not os.path.exists("checkpoints/{}".format(fruitNameA+fruitNameB))):
		os.mkdir("checkpoints/{}".format(fruitNameA+fruitNameB))
	if (not os.path.exists("checkpoints/{}/epochs/".format(fruitNameA+fruitNameB))):
		os.mkdir("checkpoints/{}/epochs/".format(fruitNameA+fruitNameB))
	if (not os.path.exists("checkpoints/{}/outputs/".format(fruitNameA+fruitNameB))):
		os.mkdir("checkpoints/{}/outputs/".format(fruitNameA+fruitNameB))

	data_path = "fruits"
	torch.cuda.empty_cache()

	device = torch.device('cuda')

	loaderA = torch.utils.data.DataLoader(ImageData(data_path+"/{}/".format(fruitNameA),torchvision.transforms.ToTensor()), batch_size=1, shuffle=True)
	loaderB = torch.utils.data.DataLoader(ImageData(data_path+"/{}/".format(fruitNameB),torchvision.transforms.ToTensor()), batch_size=1, shuffle=True)

	Ga = Generator()
	Da = Discriminator()

	Gb = Generator()
	Db = Discriminator()

	#Send to gpu
	Ga.to(device)
	Da.to(device)
	Gb.to(device)
	Db.to(device)

	summary(Ga, input_size=(3, 220, 220))
	summary(Da, input_size=(3, 220, 220))

	gOpt = torch.optim.Adam(list(Ga.parameters())+list(Gb.parameters()),
							   lr=2e-4,
							   betas=(0.5, 0.999))
	dOpt = torch.optim.Adam(list(Da.parameters())+list(Db.parameters()),
							   lr=2e-4,
							   betas=(0.5, 0.999))

	lambdaA = 10.0
	lambdaB = 10.0 

	mse = OZLoss(nn.MSELoss())
	l1 = nn.L1Loss()

	losses = {}
	scores = {}

	last_epoch = 0

	models = sorted(os.listdir("checkpoints/{}/epochs/".format(fruitNameA+fruitNameB)))
	if len(models) > 0:
		latest_model = models[-1]
		model_path = os.path.join("checkpoints/{}/epochs/".format(fruitNameA+fruitNameB), latest_model)
		print(f'loading trained model {model_path}')

		map_location = lambda storage, loc: storage
		state_dict = torch.load(model_path, map_location=map_location)
		Ga.load_state_dict(state_dict["GaState"])
		Gb.load_state_dict(state_dict["GbState"])
		Da.load_state_dict(state_dict["DaState"])
		Db.load_state_dict(state_dict["DbState"])

		if 'gOpt' in state_dict.keys():
			gOpt.load_state_dict(state_dict["gOpt"])
			dOpt.load_state_dict(state_dict["dOpt"])
			
		last_epoch = int(model_path[-6:-4])+1
	else:
		last_epoch = 0

	for epoch in range(last_epoch,epochs):
		print("Epoch {} started".format(epoch))
		epochStart = time.time()
		for i, data in enumerate(zip(loaderA,loaderB)):
				if i%100==0:
					print("		Passed samples {} ".format(i))

				a = data[0].to(device)
				b = data[1].to(device)

				# A -> B -> A cycle
				gOpt.zero_grad()

				#Generator A's GAN Loss
				fake_b = Ga(a)    
				Dpred = Da(fake_b)
				ganLossGa = mse(Dpred, True)

				#Cycle consistency for generator A
				a_hat = Gb(fake_b)
				cycleLossA = lambdaA*l1(a_hat, a)

				identityA = Gb(a)
				identityLossA =  0.5 *  lambdaA * l1(identityA, a) 

				lossA = ganLossGa + cycleLossA + identityLossA
				lossA.backward(retain_graph=True)
				gOpt.step()

				# B -> A -> A cycle
				gOpt.zero_grad()

				
				#Generator B's GAN Loss
				fake_a = Gb(b) 
				Dpred = Db(fake_a)
				ganLossGb = mse(Dpred, True)

				#Cycle consistency for generator B
				b_hat = Ga(fake_a) 
				cycleLossB = lambdaB*l1(b_hat, b)


				identityB = Ga(b)
				identityLossB =  0.5 *  lambdaB * l1(identityB, b) 

				lossB = ganLossGb + cycleLossB + identityLossB 

				lossB.backward(retain_graph=True)
				gOpt.step()

				#Discriminator A's GAN Loss
				dOpt.zero_grad()

				ganLossDa = (
					mse(Da(b), True) + 
					mse(Da(fake_b), False)
					) * 0.5

				ganLossDa.backward()
				dOpt.step()

				#Discriminator B's GAN Loss
				dOpt.zero_grad()

				ganLossDb = ( 
					mse(Db(a), True) +
					mse(Db(fake_a), False)
					) * 0.5

				ganLossDb.backward()
				dOpt.step()

				if i % 100 == 0:
					filename ='out{}_{}.png'.format(epoch, i)
					torchvision.utils.save_image(fake_b, os.path.join("checkpoints/{}/outputs".format(fruitNameA+fruitNameB), filename), normalize=True)

				if i % 1001 == 0:
					torch.save({
						'GaState': Ga.state_dict(),
						'GbState': Gb.state_dict(),
						'DaState': Da.state_dict(),
						'DbState': Db.state_dict(),
						'gOpt': gOpt.state_dict(),
						'dOpt': dOpt.state_dict()
						}, "checkpoints/{}/epochs/state{}_0{}.pth".format(fruitNameA+fruitNameB,i,epoch))

		epochTime = int(time.time() - epochStart)
		print("Epoch time: {}".format(epochTime))

#trainOnFruitPair("Apple","Banana",20)