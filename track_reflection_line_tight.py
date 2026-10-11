import cv2
import numpy as np
import os

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

os.makedirs(OUTDIR, exist_ok=True)

# Tight ROI around the actual horizontal reflection
x1, x2 = 300, 1400
y1, y2 = 650, 780

cap = cv2.VideoCapture(VIDEO)

ok, frame = cap.read()
if not ok:
    raise RuntimeError("Could not read video")

prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Initial line position:
# Find the strongest horizontal bright ridge inside the ROI.
roi = prev[y1:y2, x1:x2].astype(np.float32)
profile = np.mean(roi, axis=1)

# Smooth vertical profile
profile = cv2.GaussianBlur(
    profile.reshape(-1,1),
    (1,11),
    0
).ravel()

initial_y = y1 + int(np.argmax(profile))

print("Initial line y:", initial_y)

positions = [initial_y]

# ---------------------------------------------------------
# Track the ridge frame by frame
# ---------------------------------------------------------

search_half_width = 18

while True:

    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[y1:y2, x1:x2].astype(np.float32)

    # Horizontal average profile.
    # This suppresses small texture fluctuations.
    profile = np.mean(roi, axis=1)

    profile = cv2.GaussianBlur(
        profile.reshape(-1,1),
        (1,9),
        0
    ).ravel()

    # Search only near the previous line position.
    prev_y = positions[-1]

    lo = max(
        y1,
        prev_y - search_half_width
    )

    hi = min(
        y2,
        prev_y + search_half_width + 1
    )

    local = profile[
        lo-y1 : hi-y1
    ]

    if len(local) == 0:
        positions.append(prev_y)
        continue

    new_y = lo + int(np.argmax(local))

    # Reject absurd single-frame jumps
    if abs(new_y - prev_y) > search_half_width:
        new_y = prev_y

    positions.append(new_y)

cap.release()

positions = np.asarray(
    positions,
    dtype=np.float64
)

print("Frames tracked:", len(positions))
print(
    "Position range:",
    positions.min(),
    "to",
    positions.max(),
    "pixels"
)

# ---------------------------------------------------------
# Remove slow/static position
# ---------------------------------------------------------

displacement = (
    positions -
    np.median(positions)
)

# Save numerical data
np.savez_compressed(
    os.path.join(
        OUTDIR,
        "reflection_line_tracked.npz"
    ),
    position=positions,
    displacement=displacement,
    fps=30.00075632158794,
    roi=np.array([
        x1, x2, y1, y2
    ])
)

# ---------------------------------------------------------
# Plot displacement
# ---------------------------------------------------------

import matplotlib.pyplot as plt

t = np.arange(
    len(displacement)
) / 30.00075632158794

plt.figure(figsize=(12,4))
plt.plot(t, displacement)
plt.xlabel("Time (s)")
plt.ylabel("Line displacement (pixels)")
plt.title("Reflected-light line displacement")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "reflection_line_displacement.png"
    ),
    dpi=150
)

plt.close()

# ---------------------------------------------------------
# Overlay tracked line on middle frame
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO)

mid = len(positions) // 2

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    mid
)

ok, frame = cap.read()
cap.release()

if ok:

    y = int(round(positions[mid]))

    cv2.line(
        frame,
        (x1, y),
        (x2, y),
        (0, 0, 255),
        3
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.imwrite(
        os.path.join(
            OUTDIR,
            "reflection_line_tight_tracking.png"
        ),
        frame
    )

print()
print("Saved:")
print(
    os.path.join(
        OUTDIR,
        "reflection_line_tight_tracking.png"
    )
)
print(
    os.path.join(
        OUTDIR,
        "reflection_line_displacement.png"
    )
)
print(
    os.path.join(
        OUTDIR,
        "reflection_line_tracked.npz"
    )
)
