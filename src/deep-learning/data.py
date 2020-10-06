from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize

import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
from PIL import ImageFilter
from random import randint

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, crop_size, upscale_factor, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]		
        self.target_transform = target_transform
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
		
    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        ran1 = randint(0,2)
        if (ran1==0):
            self.input_transform = input_transform_1(self.crop_size, self.upscale_factor)
        if (ran1==1):
            self.input_transform = input_transform_2(self.crop_size, self.upscale_factor)
        if (ran1==2):
            self.input_transform = input_transform_3(self.crop_size, self.upscale_factor)      
        if self.input_transform:
            #ran2 = randint(0,0.1)
            #input = input.filter(ImageFilter.GaussianBlur(radius=ran2))
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.image_filenames)

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform_1(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor,Image.NEAREST),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

def input_transform_2(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor,Image.BICUBIC),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

def input_transform_3(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor,Image.BILINEAR),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
		
def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])


def get_training_set(upscale_factor):
    train_dir = "Drive/nn/sample"
    crop_size = calculate_valid_crop_size(128, upscale_factor)

    return DatasetFromFolder(train_dir, crop_size, upscale_factor, target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    test_dir = "Drive/nn/test"
    crop_size = calculate_valid_crop_size(128, upscale_factor)

    return DatasetFromFolder(test_dir, crop_size, upscale_factor, target_transform=target_transform(crop_size))
