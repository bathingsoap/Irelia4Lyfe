'''
Takes a single photo and returns the matrix value of it
'''
import cv2
import numpy as np

def read_face(face_file: str)-> [[int]]:
    """takes a filename containing a photo and returns its array of 
    grayscale values"""
    img = cv2.imread(face_file)
    return img


print(read_face("Faces/Snapchat-1025800267.jpg"))