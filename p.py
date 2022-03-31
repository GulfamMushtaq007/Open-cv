# how to detect face in imgs

import cv2 as cv
import numpy as np


face_cascade = cv.CascadeClassifier('C:/Users/Gulfam/Desktop/Data Sci/opencv/haarcascade_frontalface_default.xml')

img = cv.imread('860_main_beauty.png')

#img = cv.resize(img,(683,384))

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, 1.1,4)

# draw a rectangle
for (x,y,w,h) in faces:
    cv.rectangle(img, (x,y), (x+w, y+h),(255,0,0),2)


cv.imshow('img',img)
cv.waitKey(0)
cv.imwrite('facedetection.png',img)
cv.destroyAllWindows()
     










