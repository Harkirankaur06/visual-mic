import cv2
import numpy as np
import os

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUT = r"C:\Users\Harkiran\Downloads\newlightvibeexp\light_coordinate_grid.png"

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 119)

ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read frame")

# Horizontal grid every 25 pixels
for y in range(0, frame.shape[0], 25):
    cv2.line(frame, (0,y), (frame.shape[1],y), (0,0,255), 1)

    cv2.putText(
        frame,
        str(y),
        (10, max(15,y-3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0,0,255),
        1,
        cv2.LINE_AA
    )

# Vertical grid every 50 pixels
for x in range(0, frame.shape[1], 50):
    cv2.line(frame, (x,0), (x,frame.shape[0]), (255,0,0), 1)

    cv2.putText(
        frame,
        str(x),
        (x+2,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255,0,0),
        1,
        cv2.LINE_AA
    )

cv2.imwrite(OUT, frame)

print("Saved:", OUT)
