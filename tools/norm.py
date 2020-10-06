import cv2
import dlib
import numpy
import os

predicterPath = './landmarks/shape_predictor_68_face_landmarks.dat'
faceInPath = './imagesInput/'
faceOutPath = './imagesOutput/'

imageFiles = []

for file in os.listdir(faceInPath):
        if file.endswith('.jpg'):
                imageFiles.append(file)
	
for imageFile in imageFiles:
        myImage = faceInPath + imageFile
        detector = dlib.get_frontal_face_detector()
        myPredictor = dlib.shape_predictor(predicterPath)
        im = cv2.imread(myImage)
        im_rgb = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        allFaces = detector(im_rgb,1)
        if len(allFaces)==0:
                continue;
        keyPoints = dlib.full_object_detections()
        for face in allFaces:
                keyPoints.append(myPredictor(im_rgb, face))
        images = dlib.get_face_chips(im_rgb, keyPoints, size = 140)
        imageCount = 0
        for image in images:
                imageCount += 1
                currentImage = numpy.array(image).astype(numpy.uint8)
                currentImage_bgr = cv2.cvtColor(currentImage, cv2.COLOR_RGB2BGR)
                cropImage = currentImage_bgr[10:130,10:130]
                cv2.imwrite(faceOutPath+imageFile,cropImage)
        print(imageFile, 'is generated')
	

