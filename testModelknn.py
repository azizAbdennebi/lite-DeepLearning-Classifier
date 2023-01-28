import numpy as np
import cv2
import os
import pickle
filename = 'knnClassification10_model.sav'
# load the model from disk
knn= pickle.load(open(filename, 'rb'))
# testing for custom input
image=cv2.imread("auto.jpg")
image=cv2.resize(image,(32,32))
nx,ny,nrgb=image.shape
image2=image.reshape(1,(nx*ny*nrgb))
classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
result=knn.predict(image2)
print(result)
print(classes[result[0]])
