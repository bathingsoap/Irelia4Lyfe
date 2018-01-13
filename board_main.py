#!/usr/bin/python
import cv2
import sys
import os
import time
import numpy as np
from PIL import Image
import logging as log
import datetime as dt
from time import sleep

Tstart = time.time()
index= 0

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
# anterior = 0

subjects = ["", "1111", "2222", "3333", "4444", "5555", "6666" ] 
# 1111 = Anna, 2222 = Chris, 3333 = Marsh, 4444 = Kelly, 5555 = Alex, 6666 = Jerry

import mraa
import time
# Refer to the pin-out diagram for the GPIO number to silk print mapping
# in this example the number 2 maps to P10 on LinkIt Smart 7688 board
#from mraa import Result as r

rowVal = 0
colVal = 0
value = ""
pin1 = mraa.Gpio(24)
pin2 = mraa.Gpio(26)
pin3 = mraa.Gpio(30)
pin4 = mraa.Gpio(32)
pin5 = mraa.Gpio(34)
pin6 = mraa.Gpio(23)
pin7 = mraa.Gpio(25)
pin8 = mraa.Gpio(27)

pin1.dir(mraa.DIR_IN)
pin2.dir(mraa.DIR_IN)
pin3.dir(mraa.DIR_IN)
pin4.dir(mraa.DIR_IN)
pin5.dir(mraa.DIR_OUT)
pin6.dir(mraa.DIR_OUT)
pin7.dir(mraa.DIR_OUT)
pin8.dir(mraa.DIR_OUT)

if(pin1 == None):
    print('Error')
#ret = pin1.dir(mraa.DIR_IN)

#if(ret!=mraa.SUCCESS):
#    print('s')

#pin8.write(1)
#pin1.write(1)
pin8.write(1)
pin7.write(1)
pin6.write(1)
pin5.write(1)
pin4.write(0)
pin3.write(0)
pin2.write(0)
pin1.write(0)

def getID():
    result = ""
    while True:
        if(pin1.read() == 1):
            rowVal=1
        if(pin2.read() == 1):
            rowVal=2
        if(pin3.read() == 1):
            rowVal=3
        if(pin4.read() == 1):
            rowVal=4
        if not rowVal == 0:
            #num = pin.read()
            pin5.dir(mraa.DIR_IN)
            pin6.dir(mraa.DIR_IN)
            pin7.dir(mraa.DIR_IN)
            pin8.dir(mraa.DIR_IN)
            pin5.write(0)
            pin6.write(0)
            pin7.write(0)
            pin8.write(0)
            if rowVal==1:
                pin1.dir(mraa.DIR_OUT)
        if(pin3.read() == 1):
            rowVal=3
        if(pin4.read() == 1):
            rowVal=4
        if not rowVal == 0:
            #num = pin.read()
            pin5.dir(mraa.DIR_IN)
            pin6.dir(mraa.DIR_IN)
            pin7.dir(mraa.DIR_IN)
            pin8.dir(mraa.DIR_IN)
            pin5.write(0)
            pin6.write(0)
            pin7.write(0)
            pin8.write(0)
            if rowVal==1:
                pin1.dir(mraa.DIR_OUT)
                pin1.write(1)
            if rowVal==2:
                pin2.dir(mraa.DIR_OUT)
                pin2.write(1)
            if rowVal==3:
                pin3.dir(mraa.DIR_OUT)
                pin3.write(1)
            if rowVal==4:
                pin4.dir(mraa.DIR_OUT)
                pin4.write(1)
            if(pin5.read()==1):
                colVal = 4
            if(pin6.read()==1):
                colVal = 3
            if(pin7.read()==1):
                colVal = 2
            if(pin8.read()==1):
                colVal = 1
            #print (str(rowVal) + "   " +str(colVal))
    
        pin1.dir(mraa.DIR_IN)
        pin2.dir(mraa.DIR_IN)
        pin3.dir(mraa.DIR_IN)
        pin4.dir(mraa.DIR_IN)
        pin5.dir(mraa.DIR_OUT)
        pin6.dir(mraa.DIR_OUT)
        pin7.dir(mraa.DIR_OUT)
        pin8.dir(mraa.DIR_OUT)
        pin8.write(1)
        pin7.write(1)
        pin6.write(1)
        pin5.write(1)
        pin4.write(0)
        pin3.write(0)
        pin2.write(0)
        pin1.write(0)
    
        if(rowVal==1 and colVal ==1):
            value = "A"
        if(rowVal==2 and colVal ==1):
            value = "3"
        if(rowVal==3 and colVal ==1):
            value = "2"
        if(rowVal==4 and colVal ==1):
            value = "1"
        if(rowVal==1 and colVal ==2):
            value = "B"
        if(rowVal==2 and colVal ==2):
            value = "6"
        if(rowVal==3 and colVal ==2):
            value = "5"
        if(rowVal==4 and colVal ==2):
            value = "4"
        if(rowVal==1 and colVal ==3):
            value = "C"
        if(rowVal==2 and colVal ==3):
            value = "9"
        if(rowVal==3 and colVal ==3):
            value = "8"
        if(rowVal==4 and colVal ==3):
            value = "7"
        if(rowVal==1 and colVal ==4):
            value = "D"
        if(rowVal==2 and colVal ==4):
            value = "#"
        if(rowVal==3 and colVal ==4):
            value = "0"
        if(rowVal==4 and colVal ==4):
            value = "*"
            break
    
        if not value == "":
            print(value)
        result+=value
        time.sleep(0.3)
        colVal=0
        rowVal=0
        value=""
    
    return result

def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30));
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

#data will be in two lists of same size
#one list will contain all the faces
#and the other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print(faces)
print(labels)
print("Data prepared")
 
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.createLBPHFaceRecognizer()
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
def predict(test_img):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img) 
    
    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    print("Confidence", confidence)  #confirm with marshall later what confidence means
    
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    
    #draw name of predicted person if confidence > 50
    if confidence < 70:
        draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img, label_text, confidence


exists = False

while True:
    while not exists:
        IDNum = getID()
        dirs = os.listdir("Faces")
        for face in dirs:
            if IDNum in face:
                exists =True
        if not exists:
            print("Invalid ID Number")
    try:
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
        if faces == ():
            continue
        x1=0
        y1=0
        x2=0
        y2=0
        
        if len(faces) > 1:
        # If more than 1 face is detected in the frame, draw a blue rectangle
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #         if anterior != len(faces):
    #             anterior = len(faces)
    #             log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
        elif len(faces) == 0:
            continue
        
        else:
        #If only 1 face is detected in the frame, draw a green rectangle around the face
            for (x, y, w, h) in faces:
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                x1=x
                y1=y
                x2=x+w
                y2=y+h
                
    #         if anterior != len(faces):
    #             anterior = len(faces)
    #             log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
            Ttemp = time.time()
            if Ttemp - Tstart > 0.8:

                cv2.imwrite("oripics\\"+str(index)+"s.jpg",frame)
                box = (x1*0.97,y1*0.97,x2*1.03,y2*1.03)
                im = Image.open("oripics\\"+str(index)+"s.jpg")
                region = im.crop(box)
                region.save("editpics\\"+str(index)+"c.png","PNG")
                
                Tstart = Ttemp
            
                print("Predicting images...")
    
                #load comparing images
                test_img1 = cv2.imread("oripics/"+str(index)+"s.jpg")
                
                
                #perform a prediction
                print("Test image", test_img1)
                predicted_img1, name_of_person, confidence_num = predict(test_img1) #added label
                print(predicted_img1, name_of_person)
                
    #             if(predict == 0):
    #                 print("Prediction failed, retrying")
    #                 continue
                
                print("Prediction complete")
                
                #display both images
                #cv2.imshow("test",cv2.resize(predicted_img1, (400, 500)))
                
                
                index = index + 1
    
    
        # Display the resulting frame
        #cv2.imshow('Video', frame)
    
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        # Display the resulting frame
        #cv2.imshow('Video', frame)
        if(confidence_num < 50):
            if(name_of_person == IDNum):
                print("Matched ID and Face")
                exists = False
                sleep(5)
                continue
            else:
                print("ID and Face not matching")
                exists = False
                sleep(5)
                continue
        
    except cv2.error:
        print("No match, try again")
    #except OpenCV Error:
        

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
