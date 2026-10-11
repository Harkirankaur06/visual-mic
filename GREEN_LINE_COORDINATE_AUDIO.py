import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os

# ============================================================
# PATHS
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\newlightvibeexp\MANUAL_LINE_tracking.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# ACTUAL GREEN LINE REGION
# ============================================================

X1 = 700
X2 = 1050

Y1 = 750
Y2 = 900

# ============================================================
# SETTINGS
# ============================================================

AUDIO_FS = 48000

# Carrier ONLY for making the low-frequency measured movement
# audible. It does not claim to be recovered audio.
CARRIER = 440.0

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Frames:", nframes)

# ============================================================
# TRACK GREEN LINE
# ============================================================

line_y = []
line_x = []

# Also store the entire green-line shape
shape_matrix = []

for frame_no in range(nframes):

    ret, frame = cap.read()

    if not ret:
        break

    roi = frame[Y1:Y2, X1:X2]

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)

    # Green overlay detection
    mask = (
        (g > 100) &
        (g > r + 40) &
        (g > b + 40)
    )

    ys, xs = np.where(mask)

    if len(ys) < 10:

        line_y.append(np.nan)
        line_x.append(np.nan)

        shape_matrix.append(
            np.full(X2 - X1, np.nan)
        )

        continue

    # --------------------------------------------------------
    # Per-X green-line coordinate
    # --------------------------------------------------------

    y_profile = np.full(X2 - X1, np.nan)

    for x in np.unique(xs):

        yy = ys[xs == x]

        if len(yy):
            y_profile[x] = np.median(yy) + Y1

    # Fill gaps along the line
    good = np.isfinite(y_profile)

    if np.sum(good) >= 2:

        xx = np.arange(len(y_profile))

        y_profile = np.interp(
            xx,
            xx[good],
            y_profile[good]
        )

    shape_matrix.append(y_profile)

    # --------------------------------------------------------
    # Overall line coordinate
    # --------------------------------------------------------

    line_y.append(np.median(ys) + Y1)
    line_x.append(np.median(xs) + X1)

cap.release()

line_y = np.asarray(line_y, dtype=float)
line_x = np.asarray(line_x, dtype=float)
shape_matrix = np.asarray(shape_matrix, dtype=float)

# ============================================================
# FILL MISSING FRAME VALUES
# ============================================================

frames = np.arange(len(line_y))

good = np.isfinite(line_y)

line_y = np.interp(
    frames,
    frames[good],
    line_y[good]
)

line_x = np.interp(
    frames,
    frames[np.isfinite(line_x)],
    line_x[np.isfinite(line_x)]
)

print()
print("GREEN LINE Y:")
print("min:", line_y.min())
print("max:", line_y.max())
print("mean:", line_y.mean())
print("std:", line_y.std())

# ============================================================
# VERTICAL MOVEMENT
# ============================================================

movement = line_y - np.mean(line_y)

time = np.arange(len(movement)) / fps

# ============================================================
# SAVE COORDINATES
# ============================================================

np.savez(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_coordinates.npz"
    ),
    time=time,
    x=line_x,
    y=line_y,
    movement=movement,
    shape=shape_matrix,
    fps=fps
)

# ============================================================
# PLOT 1: ACTUAL GREEN LINE Y COORDINATE
# ============================================================

plt.figure(figsize=(15, 5))

plt.plot(
    time,
    line_y,
    linewidth=2
)

plt.xlabel("Time (seconds)")
plt.ylabel("Y coordinate (pixels)")

plt.title(
    "Green Line Vertical Coordinate"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_Y_COORDINATE.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# PLOT 2: MOVEMENT
# ============================================================

plt.figure(figsize=(15, 5))

plt.plot(
    time,
    movement,
    linewidth=1.5
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("Time (seconds)")
plt.ylabel("Vertical displacement (pixels)")

plt.title(
    "Measured Green Line Vertical Movement"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_MOVEMENT.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# PLOT 3: X MOVEMENT
# ============================================================

plt.figure(figsize=(15, 5))

plt.plot(
    time,
    line_x,
    linewidth=1.5
)

plt.xlabel("Time (seconds)")
plt.ylabel("X coordinate (pixels)")

plt.title(
    "Green Line Horizontal Coordinate"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_X_COORDINATE.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# SONIFICATION
#
# We preserve the measured movement as the amplitude envelope
# of a 440-Hz tone so that the vibration becomes audible.
# ============================================================

duration = time[-1]

audio_time = np.arange(
    0,
    duration,
    1.0 / AUDIO_FS
)

# Interpolate measured movement to audio sampling rate
movement_audio = np.interp(
    audio_time,
    time,
    movement
)

# Normalize movement
movement_audio -= np.mean(movement_audio)

scale = np.max(
    np.abs(movement_audio)
)

if scale > 0:
    movement_audio /= scale

# ------------------------------------------------------------
# Amplitude modulation
# ------------------------------------------------------------

envelope = 0.5 + 0.5 * movement_audio

carrier = np.sin(
    2 * np.pi * CARRIER * audio_time
)

audio = envelope * carrier

# Normalize
audio /= np.max(np.abs(audio))

audio *= 0.95

# ============================================================
# SAVE SONIFICATION
# ============================================================

wav_path = os.path.join(
    OUTDIR,
    "GREEN_LINE_VIBRATION_SONIFICATION.wav"
)

write(
    wav_path,
    AUDIO_FS,
    np.int16(audio * 32767)
)

print()
print("FILES CREATED:")
print(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_Y_COORDINATE.png"
    )
)

print(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_MOVEMENT.png"
    )
)

print(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_X_COORDINATE.png"
    )
)

print(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_coordinates.npz"
    )
)

print(wav_path)

print()
print("DONE")