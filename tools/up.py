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
        im2 = cv2.resize(im, (0,0), fx=8, fy=8,interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(faceOutPath+imageFile,im2)
        print(imageFile, 'is generated')


    
