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

ok, frame = cap.read()
if not ok:
    raise RuntimeError("Could not read video")

prev = cv2.cvtColor(
    frame[y1:y2, x1:x2],
    cv2.COLOR_BGR2GRAY
)

motion_frames = []
flow_magnitudes = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(
        frame[y1:y2, x1:x2],
        cv2.COLOR_BGR2GRAY
    )

    flow = cv2.calcOpticalFlowFarneback(
        prev,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    fx = flow[..., 0]
    fy = flow[..., 1]

    mag = np.sqrt(fx**2 + fy**2)

    motion_frames.append(flow)
    flow_magnitudes.append(mag)

    prev = gray

cap.release()

flows = np.stack(motion_frames)
mags = np.stack(flow_magnitudes)

print("Flow frames:", flows.shape)
print("Mean flow magnitude:", np.mean(mags))
print("Max flow magnitude:", np.max(mags))

# --------------------------------------------------
# 1. Average motion magnitude map
# --------------------------------------------------

mean_motion = np.mean(mags, axis=0)

motion_norm = cv2.normalize(
    mean_motion,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(
    os.path.join(
        OUTDIR,
        "optical_flow_mean_motion.png"
    ),
    motion_norm
)

# --------------------------------------------------
# 2. Temporal variability of optical flow
# --------------------------------------------------

flow_std = np.std(flows, axis=0)
flow_std_mag = np.sqrt(
    flow_std[...,0]**2 +
    flow_std[...,1]**2
)

std_norm = cv2.normalize(
    flow_std_mag,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(
    os.path.join(
        OUTDIR,
        "optical_flow_temporal_variation.png"
    ),
    std_norm
)

# --------------------------------------------------
# 3. Save flow data
# --------------------------------------------------

np.savez_compressed(
    os.path.join(
        OUTDIR,
        "optical_flow_data.npz"
    ),
    flow=flows,
    magnitude=mags,
    fps=30.0,
    x1=x1,
    x2=x2,
    y1=y1,
    y2=y2
)

print()
print("Saved:")
print("optical_flow_mean_motion.png")
print("optical_flow_temporal_variation.png")
print("optical_flow_data.npz")
