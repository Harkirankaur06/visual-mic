import cv2

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUT = r"C:\Users\Harkiran\OneDrive\visual mic\new_vid_roi.mp4"

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)

x1, x2 = 0, 1080
y1, y2 = 1100, 1920

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUT,
    fourcc,
    fps,
    (x2-x1, y2-y1)
)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    roi = frame[y1:y2, x1:x2]
    writer.write(roi)

cap.release()
writer.release()

print("Saved:", OUT)
print("ROI:", x1, y1, x2, y2)
print("Size:", x2-x1, "x", y2-y1)
