import cv2 as cv 
import numpy as np 

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')


while(True):
    _, frame = cap.read()
    gray_face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)    
    cv.imshow("Faces", frame)
    print(len(faces))
    k = cv.waitKey(1) & 0xFF 
    if k == ord('q'):
        break

cap.release()
cv.destroyAllWindows()