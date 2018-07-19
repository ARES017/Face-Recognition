'''################################ Face Detection ################################'''

import numpy as np
import cv2

### Load OpenCV Classifier

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

### Load Image

img = cv2.imread('Training Data/4.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

### Face Detection

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print('Faces found: ', len(faces))

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   ### Draw rectangles around face
    
#    roi_gray = gray[y:y+h, x:x+w]   ### Cropping the rectangele area
#    roi_color = img[y:y+h, x:x+w]   ### Cropping the rectangele area
    
### Eye Detection
    
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
cv2.imshow('img',img)  ### Display Image

 
