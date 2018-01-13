import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

###
import time
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
###
Tstart = time.time()
index= 0
###
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    x1=0
    y1=0
    x2=0
    y2=0
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        x1=x
        y1=y
        x2=x+w
        y2=y+h

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)
    
###
#    Ttemp = time.time()
#   if Ttemp - Tstart > 2:
#       cv2.imwrite(str(index)+"s.jpg",frame)
#       index = index + 1
#       Tstart = Ttemp
###
    Ttemp = time.time()
    if Ttemp - Tstart > 0.8:
        cv2.imwrite("oripics\\"+str(index)+"s.jpg",frame)
        box = (x1*0.97,y1*0.97,x2*1.03,y2*1.03)
        im = Image.open("oripics\\"+str(index)+"s.jpg")
        region = im.crop(box)
        region.save("editpics\\"+str(index)+"c.png","PNG")
        index = index + 1
        Tstart = Ttemp
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
