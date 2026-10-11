import cv2
import numpy as np
import os

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\light vibe exp"

os.makedirs(OUTDIR, exist_ok=True)

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

# Mean brightness and temporal vibration
mean_img = np.mean(V, axis=0)
std_img  = np.std(V, axis=0)

# Percentile thresholds
bright_thr = np.percentile(mean_img, 70)
motion_thr = np.percentile(std_img, 60)

# Candidate reflected-light pixels
mask = (
    (mean_img >= bright_thr) &
    (std_img >= motion_thr)
)

# Remove tiny isolated regions
mask_u8 = (mask.astype(np.uint8) * 255)

kernel = np.ones((5,5), np.uint8)
mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

# Keep reasonably large connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    mask_u8, 8
)

clean = np.zeros_like(mask_u8)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]

    if area >= 20:
        clean[labels == i] = 255

# Save mask
mask_path = os.path.join(
    OUTDIR, "reflection_candidate_mask.png"
)
cv2.imwrite(mask_path, clean)

# Create visualization over mean image
mean_u8 = cv2.normalize(
    mean_img, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

vis = cv2.cvtColor(mean_u8, cv2.COLOR_GRAY2BGR)

# Red = selected candidate reflection
vis[clean > 0] = (0, 0, 255)

vis_path = os.path.join(
    OUTDIR, "reflection_candidate_overlay.png"
)
cv2.imwrite(vis_path, vis)

print("Frames:", len(V))
print("Brightness threshold:", bright_thr)
print("Motion threshold:", motion_thr)
print("Selected pixels:", np.count_nonzero(clean))

print()
print("Saved:")
print(mask_path)
print(vis_path)
