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
# GREEN LINE ROI
# ============================================================

X1 = 700
X2 = 1050

Y1 = 750
Y2 = 900

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Frames:", nframes)

# ============================================================
# TRACK GREEN LINE FRAME BY FRAME
# ============================================================

frame_y = []
frame_x = []

for frame_no in range(nframes):

    ret, frame = cap.read()

    if not ret:
        break

    roi = frame[Y1:Y2, X1:X2]

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)

    # Detect GREEN overlay
    mask = (
        (g > 100) &
        (g > r + 40) &
        (g > b + 40)
    )

    ys, xs = np.where(mask)

    if len(ys) == 0:
        frame_y.append(np.nan)
        frame_x.append(np.nan)
        continue

    # Coordinates of green line in full image
    ys_global = ys + Y1
    xs_global = xs + X1

    # Centre coordinate of green line
    frame_y.append(np.median(ys_global))
    frame_x.append(np.median(xs_global))

cap.release()

frame_y = np.asarray(frame_y, dtype=float)
frame_x = np.asarray(frame_x, dtype=float)

# ============================================================
# FILL MISSING VALUES
# ============================================================

frames = np.arange(len(frame_y))

valid_y = np.isfinite(frame_y)
valid_x = np.isfinite(frame_x)

frame_y = np.interp(
    frames,
    frames[valid_y],
    frame_y[valid_y]
)

frame_x = np.interp(
    frames,
    frames[valid_x],
    frame_x[valid_x]
)

# ============================================================
# FRAME 0 IS THE REFERENCE
# ============================================================

y0 = frame_y[0]
x0 = frame_x[0]

# Displacement relative to FRAME 0
y_displacement = frame_y - y0
x_displacement = frame_x - x0

print()
print("FRAME 0")
print("X0:", x0)
print("Y0:", y0)

print()
print("Y DISPLACEMENT")
print("min:", y_displacement.min())
print("max:", y_displacement.max())
print("range:", y_displacement.max() - y_displacement.min())

# ============================================================
# SAVE DATA
# ============================================================

np.savez(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_FRAME_DISPLACEMENT.npz"
    ),
    frame=frames,
    time=frames / fps,
    x=frame_x,
    y=frame_y,
    x_displacement=x_displacement,
    y_displacement=y_displacement,
    fps=fps
)

# ============================================================
# PLOT EXACT FRAME-BY-FRAME Y DISPLACEMENT
# ============================================================

plt.figure(figsize=(15, 6))

plt.plot(
    frames,
    y_displacement,
    linewidth=2
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("Frame number")
plt.ylabel("Y displacement from frame 0 (pixels)")

plt.title(
    "Green Line Frame-by-Frame Vertical Displacement"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_FRAME_DISPLACEMENT.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# ALSO PLOT X DISPLACEMENT
# ============================================================

plt.figure(figsize=(15, 6))

plt.plot(
    frames,
    x_displacement,
    linewidth=2
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("Frame number")
plt.ylabel("X displacement from frame 0 (pixels)")

plt.title(
    "Green Line Frame-by-Frame Horizontal Displacement"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_X_DISPLACEMENT.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# BUILD WAVEFORM DIRECTLY FROM Y DISPLACEMENT
# ============================================================

wave = y_displacement.copy()

# Remove DC
wave -= np.mean(wave)

# Normalize
peak = np.max(np.abs(wave))

if peak > 0:
    wave = wave / peak

# ============================================================
# SAVE THE ACTUAL 30-FPS WAVEFORM
# ============================================================

write(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_FRAME_WAVE_30Hz.wav"
    ),
    int(round(fps)),
    np.int16(wave * 32767)
)

# ============================================================
# MAKE A 48 kHz VERSION OF THE SAME WAVEFORM
#
# IMPORTANT:
# This is interpolation only.
# We are NOT adding a carrier.
# ============================================================

duration = len(wave) / fps

audio_time = np.arange(
    0,
    duration,
    1.0 / 48000
)

frame_time = frames / fps

audio = np.interp(
    audio_time,
    frame_time,
    wave
)

audio -= np.mean(audio)

peak = np.max(np.abs(audio))

if peak > 0:
    audio /= peak

audio *= 0.95

write(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_FRAME_WAVE_48k.wav"
    ),
    48000,
    np.int16(audio * 32767)
)

# ============================================================
# PRINT EVERY FRAME DISPLACEMENT
# ============================================================

print()
print("FRAME-BY-FRAME DISPLACEMENT")
print("--------------------------------")

for i in range(len(frames)):

    print(
        f"Frame {i:3d} : "
        f"Y = {frame_y[i]:8.3f}   "
        f"Displacement = {y_displacement[i]:8.3f} px"
    )

print()
print("CREATED:")
print("GREEN_LINE_FRAME_DISPLACEMENT.png")
print("GREEN_LINE_X_DISPLACEMENT.png")
print("GREEN_LINE_FRAME_DISPLACEMENT.npz")
print("GREEN_LINE_FRAME_WAVE_30Hz.wav")
print("GREEN_LINE_FRAME_WAVE_48k.wav")

print()
print("DONE")