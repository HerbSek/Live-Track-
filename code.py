import cv2
import numpy as np
cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
cam = cv2.VideoCapture(0)
scale_factor = 0.7
if not cam.isOpened():
    raise IOError('Camera not found !!!')
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)
    if not ret:
        break
    load_cascade = cascade_face.detectMultiScale(frame, 1.2, 3) 
    for (w,h,i,j) in load_cascade:
        cv2.rectangle(frame, (w,h), (w+i,h+j), (255,0,0), 2) 
    cv2.imshow('output', frame)
    c = cv2.waitKey(1)
    if c == ord(' '):
        break
    else:
        continue
cam.release()
cv2.destroyAllWindows()




