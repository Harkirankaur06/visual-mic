import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")
OUT.mkdir(parents=True, exist_ok=True)

# LOCKED ROI
X1, X2 = 350, 1439
Y1, Y2 = 775, 850

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

profiles = []
frames = {}

save_ids = [0, n//4, n//2, 3*n//4, n-1]

while True:

    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[Y1:Y2+1, X1:X2+1].astype(np.float32)

    # Smooth small-scale water texture
    smooth = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    # Vertical gradient = horizontal edges
    gy = cv2.Sobel(
        smooth,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # The reflected band has strong horizontal structure.
    strength = np.abs(gy)

    # Average along x to suppress tiny texture
    strength = cv2.GaussianBlur(
        strength,
        (15, 1),
        0
    )

    # --------------------------------------------------------
    # FIRST PASS: strongest horizontal edge
    # --------------------------------------------------------

    raw = np.argmax(
        strength,
        axis=0
    ).astype(np.float32)

    # --------------------------------------------------------
    # CONTINUITY CONSTRAINT
    #
    # Prevent sudden jumps between unrelated bright edges.
    # --------------------------------------------------------

    track = np.zeros_like(raw)

    track[0] = raw[0]

    for x in range(1, len(raw)):

        candidate = raw[x]

        previous = track[x-1]

        # Limit movement between neighboring columns
        max_jump = 3.0

        candidate = np.clip(
            candidate,
            previous - max_jump,
            previous + max_jump
        )

        track[x] = candidate

    # Strong spatial smoothing
    track = gaussian_filter1d(
        track,
        sigma=5
    )

    profiles.append(track + Y1)

    idx = len(profiles) - 1

    if idx in save_ids:
        frames[idx] = frame.copy()

cap.release()

profiles = np.asarray(profiles)

print("Profiles:", profiles.shape)
print("FPS:", fps)
print("ROI:", X1, X2, Y1, Y2)

# ------------------------------------------------------------
# REMOVE STATIC SHAPE
# ------------------------------------------------------------

reference = np.median(
    profiles,
    axis=0
)

deformation = profiles - reference[None, :]

# Remove frame-wide vertical movement
deformation -= np.median(
    deformation,
    axis=1,
    keepdims=True
)

# ------------------------------------------------------------
# OVERLAY
# ------------------------------------------------------------

idx = n // 2

overlay = frames[idx].copy()

cv2.rectangle(
    overlay,
    (X1, Y1),
    (X2, Y2),
    (0, 0, 255),
    3
)

pts = []

for x in range(profiles.shape[1]):

    xx = X1 + x

    yy = int(
        np.clip(
            profiles[idx, x],
            Y1,
            Y2
        )
    )

    pts.append((xx, yy))

pts = np.asarray(
    pts,
    dtype=np.int32
)

cv2.polylines(
    overlay,
    [pts.reshape(-1, 1, 2)],
    False,
    (0, 255, 0),
    3
)

cv2.imwrite(
    str(OUT / "reflection_continuous_overlay.png"),
    overlay
)

# ------------------------------------------------------------
# DEFORMATION MAP
# ------------------------------------------------------------

D = deformation

scale = max(
    1,
    D.shape[1] // 1000
)

D_small = D[:, ::scale]

lim = np.percentile(
    np.abs(D_small),
    98
)

if lim < 1e-6:
    lim = 1

D_vis = np.clip(
    D_small,
    -lim,
    lim
)

D_vis = (
    (D_vis + lim)
    /
    (2 * lim)
    *
    255
).astype(np.uint8)

cv2.imwrite(
    str(OUT / "reflection_continuous_deformation.png"),
    D_vis
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

np.savez(
    OUT / "reflection_continuous_tracking.npz",
    profiles=profiles,
    deformation=deformation,
    fps=fps,
    roi=np.array([X1, X2, Y1, Y2])
)

print("\nDONE")
print(OUT / "reflection_continuous_overlay.png")
print(OUT / "reflection_continuous_deformation.png")
print(OUT / "reflection_continuous_tracking.npz")
