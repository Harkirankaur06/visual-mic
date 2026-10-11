import cv2
import numpy as np
import os

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\light vibe exp"

os.makedirs(OUTDIR, exist_ok=True)

# Your ROI
x1, x2 = 100, 400
y1, y2 = 700, 1000

cap = cv2.VideoCapture(VIDEO)

frames = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[y1:y2, x1:x2]

    # Edge detection
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    edges = cv2.Canny(
        roi_blur,
        30,
        100
    )

    frames.append(edges)

cap.release()

E = np.asarray(frames)

print("Frames:", E.shape)

# ---------------------------------------------------------
# Temporal edge persistence
# ---------------------------------------------------------

persistence = np.mean(E > 0, axis=0)

persistence_img = (
    persistence / np.max(persistence) * 255
).astype(np.uint8)

cv2.imwrite(
    os.path.join(
        OUTDIR,
        "water_edge_persistence.png"
    ),
    persistence_img
)

# ---------------------------------------------------------
# Create a color overlay on the mean frame
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 140)

ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read middle frame")

roi_color = frame[y1:y2, x1:x2].copy()

# Persistent edges = red
roi_color[E.mean(axis=0) > 0.12] = (0, 0, 255)

cv2.imwrite(
    os.path.join(
        OUTDIR,
        "water_edge_tracking_overlay.png"
    ),
    roi_color
)

print()
print("Saved:")
print(
    os.path.join(
        OUTDIR,
        "water_edge_persistence.png"
    )
)

print(
    os.path.join(
        OUTDIR,
        "water_edge_tracking_overlay.png"
    )
)
