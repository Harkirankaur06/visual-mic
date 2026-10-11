import cv2
import numpy as np
from pathlib import Path

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")
OUT.mkdir(parents=True, exist_ok=True)

# LOCKED TARGET REGION
X1, X2 = 350, 1500
Y1, Y2 = 740, 880

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Clamp ROI to actual video dimensions
X2 = min(X2, width)

profiles = []
saved_frames = {}

frames_to_save = [0, n//4, n//2, 3*n//4, n-1]

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[Y1:Y2, X1:X2].astype(np.float32)

    # Smooth vertically to make the bright reflection ridge clearer
    roi_s = cv2.GaussianBlur(roi, (1, 9), 0)

    # Brightest point in each column
    y_profile = np.argmax(roi_s, axis=0).astype(float) + Y1

    # Smooth the tracked line horizontally
    y_profile = cv2.GaussianBlur(
        y_profile.reshape(1, -1).astype(np.float32),
        (31, 1),
        0
    ).ravel()

    profiles.append(y_profile)

    frame_idx = len(profiles) - 1

    if frame_idx in frames_to_save:
        saved_frames[frame_idx] = frame.copy()

cap.release()

profiles = np.asarray(profiles)

print("profiles:", profiles.shape)
print("Actual ROI:", X1, X2, Y1, Y2)

# ------------------------------------------------------------
# REMOVE STATIC SHAPE
# ------------------------------------------------------------

static_shape = np.median(profiles, axis=0)

deformation = profiles - static_shape

# ------------------------------------------------------------
# TRACKING OVERLAY
# ------------------------------------------------------------

idx = min(n // 2, len(profiles) - 1)

frame = saved_frames[idx].copy()

cv2.rectangle(
    frame,
    (X1, Y1),
    (X2 - 1, Y2 - 1),
    (0, 0, 255),
    3
)

# IMPORTANT:
# Use actual profile width, NOT X2-X1,
# because X2 was clipped to video width.

num_cols = profiles.shape[1]

pts = []

for x in range(num_cols):
    xx = X1 + x

    if xx >= frame.shape[1]:
        break

    yy = int(round(profiles[idx, x]))

    yy = max(Y1, min(Y2 - 1, yy))

    pts.append((xx, yy))

pts = np.asarray(pts, dtype=np.int32)

if len(pts) > 1:
    cv2.polylines(
        frame,
        [pts.reshape(-1, 1, 2)],
        False,
        (0, 255, 0),
        2
    )

cv2.imwrite(
    str(OUT / "reflection_shape_overlay.png"),
    frame
)

# ------------------------------------------------------------
# SHAPE DEFORMATION MAP
# ------------------------------------------------------------

D = deformation

scale = max(1, D.shape[1] // 1000)

D_small = D[:, ::scale]

lim = np.percentile(np.abs(D_small), 98)

if lim == 0:
    lim = 1

D_vis = np.clip(D_small, -lim, lim)

D_vis = (
    (D_vis + lim) /
    (2 * lim) *
    255
).astype(np.uint8)

cv2.imwrite(
    str(OUT / "reflection_shape_deformation.png"),
    D_vis
)

# ------------------------------------------------------------
# TEMPORAL ENERGY
# ------------------------------------------------------------

energy = np.std(deformation, axis=1)

energy_img = energy - energy.min()

if energy_img.max() > 0:
    energy_img /= energy_img.max()

energy_img = (energy_img * 255).astype(np.uint8)

energy_img = cv2.resize(
    energy_img.reshape(-1, 1),
    (1000, len(energy_img)),
    interpolation=cv2.INTER_NEAREST
)

cv2.imwrite(
    str(OUT / "reflection_temporal_energy.png"),
    energy_img
)

# ------------------------------------------------------------
# NUMERICAL SUMMARY
# ------------------------------------------------------------

print("\nReflection deformation:")
print("Mean abs displacement:",
      float(np.mean(np.abs(deformation))))

print("Std displacement:",
      float(np.std(deformation)))

print("95th percentile:",
      float(np.percentile(np.abs(deformation), 95)))

print("\nSaved:")
print(OUT / "reflection_shape_overlay.png")
print(OUT / "reflection_shape_deformation.png")
print(OUT / "reflection_temporal_energy.png")
