import numpy as np
import cv2
import imutils
import datetime


gun_cascade = cv2.CascadeClassifier("cascade.xml")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
firstFrame = None
Gun_exist = None
while True:
    ret, frame = camera.read()


    ret, frame = camera.read()
    if not ret or frame is None:
        print("Warning: Could not read from camera!")
        continue  # Skip the iteration instead of processing a None frame

    frame = imutils.resize(frame, width=500)



    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    if len(gun) > 0:
        Gun_exist = True

    for x, y, w, h in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_colour = frame[y : y + h, x : x + w]
    if firstFrame is None:
        firstFrame = gray
        continue
    cv2.imshow("Gun Detection", frame)
    key=cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    if Gun_exist:
        print("Gun Detected")
    else:
        print("No Gun Detected")
camera.release()
cv2.destroyAllWindows()