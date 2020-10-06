import cv2
import dlib
import numpy as np
import os
from PIL import Image
import PIL
from PIL import ImageFilter
from random import randint
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.init as init
import random

faceInPath = './imagesInput/'
faceOutPath = './imagesOutput/'

imageFiles = []

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        return din + torch.autograd.Variable(torch.randn(din.size()) * self.stddev)


for file in os.listdir(faceInPath):
        if file.endswith('.jpg'):
                imageFiles.append(file)
	
for imageFile in imageFiles:
        myImage = faceInPath + imageFile
        im = Image.open(myImage)
        #rad = randint(1,3)
        #im = im.filter(ImageFilter.GaussianBlur(0.75))
        im = im.resize((32,32),PIL.Image.BICUBIC)
        #img_to_tensor = ToTensor()
        #out = img_to_tensor(im).view(-1, 3, im.size[1], im.size[0])
        #sd = random.uniform(0.01,0.03)
        #noise = GaussianNoise(0.005)
        #out = noise(out)
        #out = torch.clamp(out,0,1)
        ##out = out[0].mul(255).byte()
       # npimg = np.transpose(out.numpy(),(1,2,0))
        #out_ycbcr = Image.fromarray(npimg, mode='RGB')

        #out_ycbcr.save(faceOutPath+imageFile,"PNG")
        im.save(faceOutPath+imageFile,"PNG")
        print(imageFile, 'is generated')


    
