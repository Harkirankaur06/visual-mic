import cv2
import numpy as np
import os

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\light vibe exp"

os.makedirs(OUTDIR, exist_ok=True)

# Reflected-light ROI
x1, x2 = 100, 400
y1, y2 = 700, 1000

cap = cv2.VideoCapture(VIDEO)

frames = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    frames.append(gray.astype(np.float32))

cap.release()

V = np.stack(frames)

print("Frames:", V.shape[0])
print("ROI:", V.shape[1], "x", V.shape[2])

# Temporal standard deviation at every pixel
std_map = np.std(V, axis=0)

# Save normalized motion map
norm = cv2.normalize(
    std_map, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(
    os.path.join(OUTDIR, "reflection_temporal_motion.png"),
    norm
)

# Mean image
mean_img = np.mean(V, axis=0)
mean_norm = cv2.normalize(
    mean_img, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(
    os.path.join(OUTDIR, "reflection_mean.png"),
    mean_norm
)

# Save raw data
np.savez_compressed(
    os.path.join(OUTDIR, "reflection_roi_video.npz"),
    video=V,
    x1=x1, x2=x2,
    y1=y1, y2=y2,
    fps=30.0
)

print()
print("Saved:")
print(os.path.join(OUTDIR, "reflection_temporal_motion.png"))
print(os.path.join(OUTDIR, "reflection_mean.png"))
print(os.path.join(OUTDIR, "reflection_roi_video.npz"))
