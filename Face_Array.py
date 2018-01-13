'''
Takes a single photo and returns the matrix value of it
'''
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(type(face_cascade))

def read_face(face_file: str)-> [[int]]:
    """takes a filename containing a photo and returns its array of 
    grayscale values"""
    img = cv2.imread(face_file) #load image
    resized_image = cv2.resize(img, (450, 800)) 
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) #grayscale image
    #print(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #finding face, returns rectangles
    print(faces)
    f = faces[0]
    x, y, w, h = f
    print(x, y, w, h)
    cropped = resized_image[y:y+h, x:x+w]

    cv2.imshow(face_file,resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow(face_file,cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped

if __name__ == "__main__":
    print(read_face("Faces/Snapchat-1025800267.jpg"))
    
    
    
    