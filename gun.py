import numpy as np
import cv2
import imutils
import datetime
import tkinter as tk
from tkinter import messagebox
import time

# Load your Haar cascade file
gun_cascade = cv2.CascadeClassifier("cascade.xml")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

firstFrame = None
detection_count = 0
no_detection_count = 0
Gun_exist = False
alert_cooldown = 5  # seconds
last_alert_time = 0  # store the last time an alert was shown

DETECTION_THRESHOLD = 5     # number of frames in which gun is detected
NO_DETECTION_THRESHOLD = 5  # number of frames in which gun is not detected

def show_alert():
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Alert!", "Gun Detected! Click OK to resume scanning.")
    root.destroy()

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Warning: Could not read from camera!")
        continue

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    detected = len(gun) > 0

    for x, y, w, h in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Update detection counters
    if detected:
        detection_count += 1
        no_detection_count = 0
    else:
        detection_count = 0
        no_detection_count += 1

    # Trigger alert only if detection is stable over several frames
    current_time = time.time()
    if detection_count >= DETECTION_THRESHOLD and not Gun_exist and (current_time - last_alert_time) > alert_cooldown:
        show_alert()
        Gun_exist = True
        last_alert_time = current_time

    # Reset flag if no gun is detected for a while
    if no_detection_count >= NO_DETECTION_THRESHOLD:
        Gun_exist = False

    cv2.imshow("Gun Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
