#!/usr/bin/python3
from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
from model import Generator_8x, Generator_4x

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

		
def resolver(myModel):
	unnormalizer = Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])
	normalizer = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
	imageFiles = []
	for file in os.listdir("./test_lr/"):
			if file.endswith('.jpg'):
					imageFiles.append(file)
					
	for imageFile in imageFiles:
		myImage = "./test_lr/" + imageFile
		img = Image.open(myImage)
		y, cb, cr = img.split()

		img_to_tensor = ToTensor()
		input = normalizer(img_to_tensor(img)).view(-1, 3, img.size[1], img.size[0])

		#myModel = myModel.cuda()
		#input = input.cuda()	
		
		out = myModel(input)
		out = unnormalizer(out[0])
		out = out.cpu()
		out = torch.clamp(out,0,1)
		out = out.mul(255).byte()
		npimg = np.transpose(out.numpy(),(1,2,0))
		out_ycbcr = Image.fromarray(npimg, mode='RGB')

		#plt.imshow(out[0][0].detach().numpy(),cmap='gray')
		#plt.show()
	#	torchvision.utils.save_image(out, "./test_hr/" + imageFile)
	#	out_img_y = out.data.numpy()
	#	out_img_y *= 255.0
	#	print(out_img_y.size)
	#	out_img_y = out_img_y.clip(0, 255)
#		out = F.to_pil_image(out_ycbcr)
	#	out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

		#out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
		#out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
		#out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

		out_ycbcr.save("./test_hr/" + imageFile)
		print('output image saved to ', imageFile)
		
		
myModel = Generator_4x()
myModel.load_state_dict(torch.load('./logs/test.pth', map_location=lambda storage, loc: storage))
resolver(myModel)
#
