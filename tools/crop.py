import cv2
import dlib
import numpy as np
import os

faceInPath = './imagesInput/'
faceOutPath = './imagesOutput/'

imageFiles = []

for file in os.listdir(faceInPath):
        if file.endswith('.jpg'):
                imageFiles.append(file)
	
for imageFile in imageFiles:
        myImage = faceInPath + imageFile
        im = cv2.imread(myImage)
        im2 = im[61:250-61,61:250-61]
        cv2.imwrite(faceOutPath+imageFile,im2)
        print(imageFile, 'is generated')


    
