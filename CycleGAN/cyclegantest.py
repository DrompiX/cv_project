import torch
import torch.nn as nn
import torchvision
import time
import os
from PIL import Image
from torchsummary import summary
from model import *
from classifier.model import predictFolder
import imageio

fruitsFolder = "tests"
def transferAtoB(fruitNameA,fruitNameB,amount):
	dirname = fruitNameA+fruitNameB
	useGenB = False

	if (not os.path.exists("checkpoints/{}".format(dirname))):
		dirname = fruitNameB+fruitNameA
		useGenB = True

	if (not os.path.exists("checkpoints/{}".format(dirname))):
		print(fruitNameA+" "+ fruitNameB + " transfer cannot be made")
		return 		

	if (not os.path.exists("toClassify")):
		os.mkdir("toClassify")

	if (not os.path.exists("toClassify/"+fruitNameB)):
		os.mkdir("toClassify/"+fruitNameB)

	loaderA = torch.utils.data.DataLoader(ImageData(fruitsFolder+"/{}/".format(fruitNameA),torchvision.transforms.ToTensor()), batch_size=1, shuffle=True)

	torch.cuda.empty_cache()
	device = torch.device('cuda:0')

	Ga = Generator()

	Ga.to(device)

	models = sorted(os.listdir("checkpoints/{}/epochs/".format(dirname)),key=lambda path: os.path.getctime("checkpoints/{}/epochs/".format(dirname)+path))
	if len(models) > 0:
		latest_model = models[-1]
		model_path = os.path.join("checkpoints/{}/epochs/".format(dirname), latest_model)
		print(f'loading trained model {model_path}')

		map_location = lambda storage, loc: storage
		state_dict = torch.load(model_path, map_location=map_location)
		if not useGenB:
			Ga.load_state_dict(state_dict["GaState"])
		else:
			Ga.load_state_dict(state_dict["GbState"])

	for i,img in enumerate(loaderA):
		image = img.to(device)

		transfered_image = Ga(image)

		torchvision.utils.save_image(transfered_image , "toClassify/"+fruitNameB+"/{}_{}.png".format(fruitNameA,i), normalize=True)
		if i > amount:
			break

fruits = ["Banana","Lemon","Orange","Apple","Cocos"]

for fruitA in fruits:
	for fruitB in fruits:
		if fruitA!=fruitB:
			transferAtoB(fruitA,fruitB,100)	

for fruit in fruits:
	foolMeter = 0
	preds = predictFolder("toClassify/"+fruit)
	for pred in preds:
		if pred==fruit:
			foolMeter+=1
	print("Generated {} score : {}".format(fruit,foolMeter/len(preds)))
