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

		
def resolver(input_lr,output_sr,upscale, dataset ,method):
        if(upscale==4):
                myModel = Generator_4x()
                if(method=='CNN'):
                        if(dataset==1):
                                myModel.load_state_dict(torch.load('./logs/cnn_4x_1.pth', map_location=lambda storage, loc: storage))
                        if(dataset==2):
                                myModel.load_state_dict(torch.load('./logs/cnn_4x_2.pth', map_location=lambda storage, loc: storage))
                if(method=='GAN'):
                        if(dataset==1):
                                myModel.load_state_dict(torch.load('./logs/gan_4x_1.pth', map_location=lambda storage, loc: storage))
                        if(dataset==2):
                                myModel.load_state_dict(torch.load('./logs/gan_4x_2.pth', map_location=lambda storage, loc: storage))
        if(upscale==8):
                myModel = Generator_8x()
                if(method=='CNN'):
                        if(dataset==1):
                                myModel.load_state_dict(torch.load('./logs/cnn_8x_1.pth', map_location=lambda storage, loc: storage))
                        if(dataset==2):
                                myModel.load_state_dict(torch.load('./logs/cnn_8x_2.pth', map_location=lambda storage, loc: storage))
                if(method=='GAN'):
                        if(dataset==1):
                                myModel.load_state_dict(torch.load('./logs/gan_8x_1.pth', map_location=lambda storage, loc: storage))
                        if(dataset==2):
                                myModel.load_state_dict(torch.load('./logs/gan_8x_2.pth', map_location=lambda storage, loc: storage))              

        img = Image.open(input_lr)
        
        unnormalizer = Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])
        normalizer = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        img_to_tensor = ToTensor()
        input = normalizer(img_to_tensor(img)).view(-1, 3, img.size[1], img.size[0])
			
        out = myModel(input)
        out = unnormalizer(out[0])
        out = out.cpu()
        out = torch.clamp(out,0,1)
        out = out.mul(255).byte()
        npimg = np.transpose(out.numpy(),(1,2,0))
        out_ycbcr = Image.fromarray(npimg, mode='RGB')
        out_ycbcr.save(output_sr)
        print('output image saved to ', output_sr)
		
