import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# TARGET
# ============================================================

X1, X2 = 350, 1439
Y1, Y2 = 775, 850

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ============================================================
# FIRST FRAME: ESTABLISH THE STRAIGHT EDGE
# ============================================================

ok, first = cap.read()

if not ok:
    raise RuntimeError("Could not read video")

gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

roi = gray[Y1:Y2+1, X1:X2+1].astype(np.float32)

# Heavy smoothing removes water texture.
smooth = cv2.GaussianBlur(
    roi,
    (21, 21),
    0
)

# Vertical gradient.
# We are looking specifically for the dark -> bright transition.
gy = cv2.Sobel(
    smooth,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)

# Average gradient horizontally.
# This tells us where the dominant straight horizontal boundary is.
mean_gradient = np.mean(
    gy,
    axis=1
)

# Strongest positive dark->bright transition.
baseline_local = np.argmax(
    mean_gradient
)

baseline_y = Y1 + baseline_local

print("Detected straight edge baseline:", baseline_y)

# ============================================================
# TRACK EDGE IN EACH FRAME
# ============================================================

profiles = []
frames = {}

save_ids = [
    0,
    n // 4,
    n // 2,
    3 * n // 4,
    n - 1
]

# The edge should remain close to the baseline.
SEARCH = 18

# Rewind
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:

    ok, frame = cap.read()

    if not ok:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[
        Y1:Y2+1,
        X1:X2+1
    ].astype(np.float32)

    # --------------------------------------------------------
    # Smooth enough to remove surface texture,
    # but preserve the large straight edge.
    # --------------------------------------------------------

    smooth = cv2.GaussianBlur(
        roi,
        (15, 15),
        0
    )

    gy = cv2.Sobel(
        smooth,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Positive gradient = dark -> bright.
    positive = np.maximum(
        gy,
        0
    )

    # --------------------------------------------------------
    # For every x, search ONLY around the known straight edge.
    # --------------------------------------------------------

    profile = np.zeros(
        positive.shape[1],
        dtype=np.float64
    )

    lo = max(
        0,
        baseline_local - SEARCH
    )

    hi = min(
        positive.shape[0],
        baseline_local + SEARCH + 1
    )

    for x in range(
        positive.shape[1]
    ):

        column = positive[
            lo:hi,
            x
        ]

        # Smooth vertically so tiny texture peaks don't win.
        column = gaussian_filter1d(
            column,
            sigma=2
        )

        peak = np.argmax(
            column
        )

        profile[x] = (
            lo +
            peak +
            Y1
        )

    # --------------------------------------------------------
    # Straight-edge continuity
    # --------------------------------------------------------

    profile = gaussian_filter1d(
        profile,
        sigma=8
    )

    profiles.append(
        profile
    )

    idx = len(profiles) - 1

    if idx in save_ids:
        frames[idx] = frame.copy()

cap.release()

profiles = np.asarray(
    profiles
)

print("Profiles:", profiles.shape)

# ============================================================
# REMOVE THE STATIC STRAIGHT LINE
# ============================================================

reference = np.median(
    profiles,
    axis=0
)

deformation = (
    profiles -
    reference[None, :]
)

# Remove frame-wide translation.
deformation -= np.median(
    deformation,
    axis=1,
    keepdims=True
)

# ============================================================
# MIDDLE FRAME OVERLAY
# ============================================================

idx = n // 2

overlay = frames[idx].copy()

# ROI
cv2.rectangle(
    overlay,
    (X1, Y1),
    (X2, Y2),
    (0, 0, 255),
    3
)

# Original detected straight baseline
cv2.line(
    overlay,
    (X1, baseline_y),
    (X2, baseline_y),
    (255, 0, 0),
    2
)

# Tracked edge
pts = []

for x in range(
    profiles.shape[1]
):

    xx = X1 + x

    yy = int(
        np.clip(
            profiles[idx, x],
            Y1,
            Y2
        )
    )

    pts.append(
        (xx, yy)
    )

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
    str(
        OUT /
        "straight_edge_tracking.png"
    ),
    overlay
)

# ============================================================
# EDGE DEFORMATION MAP
# ============================================================

scale = max(
    1,
    deformation.shape[1] // 1000
)

D = deformation[
    :,
    ::scale
]

lim = np.percentile(
    np.abs(D),
    98
)

if lim < 1e-6:
    lim = 1

D = np.clip(
    D,
    -lim,
    lim
)

D_img = (
    (D + lim)
    /
    (2 * lim)
    *
    255
).astype(np.uint8)

cv2.imwrite(
    str(
        OUT /
        "straight_edge_deformation.png"
    ),
    D_img
)

# ============================================================
# SAVE DATA
# ============================================================

np.savez(
    OUT /
    "straight_edge_tracking.npz",
    profiles=profiles,
    deformation=deformation,
    baseline_y=baseline_y,
    fps=fps,
    roi=np.array([
        X1,
        X2,
        Y1,
        Y2
    ])
)

print()
print("========================================")
print("DONE")
print("========================================")
print(
    OUT /
    "straight_edge_tracking.png"
)
print(
    OUT /
    "straight_edge_deformation.png"
)
print(
    OUT /
    "straight_edge_tracking.npz"
)
print()
print(
    "Mean displacement:",
    np.mean(np.abs(deformation)),
    "pixels"
)
print(
    "95% displacement:",
    np.percentile(
        np.abs(deformation),
        95
    ),
    "pixels"
)
