# Detection Framework  
# ///// cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')  //////
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

cam = cv2.VideoCapture(0)
scale_factor = 0.8


while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, None, fx = scale_factor, fy = scale_factor , interpolation = cv2.INTER_CUBIC)
    frame = cv2.flip(frame, 1)
    input1 = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_rectangle = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow('Input', input1)
    cv2.imshow('Face Detector', frame)


    c = cv2.waitKey(1)
    if c == 27:
        break

cam.release()
cv2.destroyAllWindows()

