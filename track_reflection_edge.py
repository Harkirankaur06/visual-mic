import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOCKED TARGET
# ------------------------------------------------------------

X1, X2 = 350, 1439
Y1, Y2 = 775, 850

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

edge_profiles = []
frames = {}

save_ids = [0, n//4, n//2, 3*n//4, n-1]

# ------------------------------------------------------------
# TRACK THE UPPER EDGE OF THE REFLECTED LIGHT
# ------------------------------------------------------------

while True:

    ok, frame = cap.read()

    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[Y1:Y2+1, X1:X2+1].astype(np.float32)

    # Reduce tiny water texture
    smooth = cv2.GaussianBlur(
        roi,
        (9, 9),
        0
    )

    # --------------------------------------------------------
    # VERTICAL GRADIENT
    #
    # Positive gradient means brightness increases downward.
    # This targets the upper edge of the bright reflection.
    # --------------------------------------------------------

    gy = cv2.Sobel(
        smooth,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    positive_edge = np.maximum(gy, 0)

    # Suppress tiny local texture
    positive_edge = cv2.GaussianBlur(
        positive_edge,
        (15, 1),
        0
    )

    # --------------------------------------------------------
    # For every x, choose the strongest POSITIVE edge.
    # --------------------------------------------------------

    raw = np.argmax(
        positive_edge,
        axis=0
    ).astype(np.float64)

    # --------------------------------------------------------
    # Smooth spatially across x.
    # The physical reflection edge should be continuous.
    # --------------------------------------------------------

    edge = gaussian_filter1d(
        raw,
        sigma=7
    )

    edge_profiles.append(edge + Y1)

    idx = len(edge_profiles) - 1

    if idx in save_ids:
        frames[idx] = frame.copy()

cap.release()

edge_profiles = np.asarray(edge_profiles)

print("Edge profiles:", edge_profiles.shape)
print("FPS:", fps)
print("ROI:", X1, X2, Y1, Y2)

# ------------------------------------------------------------
# REMOVE STATIC EDGE SHAPE
# ------------------------------------------------------------

reference = np.median(
    edge_profiles,
    axis=0
)

deformation = (
    edge_profiles -
    reference[None, :]
)

# Remove frame-wide camera movement
deformation -= np.median(
    deformation,
    axis=1,
    keepdims=True
)

# ------------------------------------------------------------
# DRAW MIDDLE-FRAME EDGE
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

for x in range(edge_profiles.shape[1]):

    xx = X1 + x

    yy = int(
        np.clip(
            edge_profiles[idx, x],
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
    str(OUT / "reflection_EDGE_overlay.png"),
    overlay
)

# ------------------------------------------------------------
# EDGE DEFORMATION MAP
# ------------------------------------------------------------

scale = max(
    1,
    deformation.shape[1] // 1000
)

D = deformation[:, ::scale]

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
    str(OUT / "reflection_EDGE_deformation.png"),
    D_img
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

np.savez(
    OUT / "reflection_EDGE_tracking.npz",
    edge_profiles=edge_profiles,
    deformation=deformation,
    fps=fps,
    roi=np.array([X1, X2, Y1, Y2])
)

print("\nDONE")
print(OUT / "reflection_EDGE_overlay.png")
print(OUT / "reflection_EDGE_deformation.png")
print(OUT / "reflection_EDGE_tracking.npz")

print("\nMean edge movement:",
      np.mean(np.abs(deformation)), "pixels")

print("95th percentile:",
      np.percentile(np.abs(deformation), 95), "pixels")
