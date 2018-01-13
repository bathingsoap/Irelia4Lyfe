#!/usr/bin/python
'''
Takes a single photo and returns the matrix value of it
'''
import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(type(face_cascade))

def read_face(face_file: str)-> [[int]]:
    """takes a filename containing a photo and returns its array of 
    grayscale values"""
    img = cv2.imread(face_file) #load image
    resized_image = cv2.resize(img, (450, 800)) 
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) #grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #finding face, returns rectangles
    if faces != ():
            
        f = faces[0]
        x, y, w, h = f
        cropped = resized_image[y:y+h, x:x+w]
    # 
    #     cv2.imshow(face_file,resized_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     
    #     cv2.imshow(face_file,cropped)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
            
        return cropped
    else:
        return ()

if __name__ == "__main__":
    face_list = ["Faces/Snapchat-840812430.jpg", "Faces/Snapchat-1025800267.jpg", "Faces/Snapchat-1068462290.jpg", "Faces/Snapchat-145761425.jpg", "Faces/Snapchat-1681206859.jpg", "Faces/Snapchat-78818167.jpg"]
    
    #if Cropped_images folder is not created yet, create it
    if not os.path.exists("Faces/Cropped_Images"):
        os.makedirs("Faces/Cropped_Images")
        
    for i in range(len(face_list)):
        print(i) 
        crop = read_face(face_list[i])
        
        #check if a face was detected and only write if it was
        if crop != ():
            cv2.imwrite("Faces/Cropped_Images/{}.png".format(i), crop)
    
    
    
    