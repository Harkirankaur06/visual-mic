import cv2
import numpy as np
import os

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

os.makedirs(OUTDIR, exist_ok=True)

# Broad region containing the horizontal reflected-light line.
# We deliberately keep some water above and below it.
x1, x2 = 200, 1400
y1, y2 = 500, 850

cap = cv2.VideoCapture(VIDEO)

positions = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[y1:y2, x1:x2].astype(np.float32)

    # Smooth noise while retaining the relatively thin bright line.
    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    # For every x-column, find the strongest bright horizontal
    # reflection within the expected vertical range.
    pos = np.argmax(roi, axis=0)

    # Convert ROI coordinates to full-frame y coordinates.
    pos = pos + y1

    positions.append(pos)

cap.release()

P = np.asarray(positions, dtype=np.float64)

print("Frames:", P.shape[0])
print("X samples:", P.shape[1])

# ---------------------------------------------------------
# Remove the static line shape
# ---------------------------------------------------------
# The average line location is mostly static geometry.
# We want ONLY its temporal ripple.

baseline = np.median(P, axis=0)

D = P - baseline[None, :]

# Remove frame-wide vertical camera drift.
D = D - np.median(D, axis=1, keepdims=True)

# ---------------------------------------------------------
# Save raw displacement
# ---------------------------------------------------------

np.savez_compressed(
    os.path.join(
        OUTDIR,
        "reflection_line_displacement.npz"
    ),
    displacement=D,
    position=P,
    baseline=baseline,
    fps=30.00075632158794,
    x1=x1,
    x2=x2,
    y1=y1,
    y2=y2
)

# ---------------------------------------------------------
# Visualization 1:
# displacement map
# ---------------------------------------------------------

disp = D.T

# Robust display range
lo, hi = np.percentile(disp, [2, 98])

disp_clip = np.clip(disp, lo, hi)

disp_img = cv2.normalize(
    disp_clip,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(
    os.path.join(
        OUTDIR,
        "reflection_line_ripple_map.png"
    ),
    disp_img
)

# ---------------------------------------------------------
# Visualization 2:
# overlay tracked line on middle frame
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO)
cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    P.shape[0] // 2
)

ok, frame = cap.read()
cap.release()

if ok:

    vis = frame.copy()

    frame_idx = P.shape[0] // 2

    xs = np.arange(x1, x2)
    ys = P[frame_idx].astype(int)

    valid = (
        (ys >= 0) &
        (ys < frame.shape[0])
    )

    for x, y in zip(xs[valid], ys[valid]):

        cv2.circle(
            vis,
            (int(x), int(y)),
            1,
            (0, 0, 255),
            -1
        )

    cv2.rectangle(
        vis,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.imwrite(
        os.path.join(
            OUTDIR,
            "reflection_line_tracking_overlay.png"
        ),
        vis
    )

print()
print("Saved:")
print(
    os.path.join(
        OUTDIR,
        "reflection_line_ripple_map.png"
    )
)
print(
    os.path.join(
        OUTDIR,
        "reflection_line_tracking_overlay.png"
    )
)
print(
    os.path.join(
        OUTDIR,
        "reflection_line_displacement.npz"
    )
)
