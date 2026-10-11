import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import welch
import os

# ============================================================
# PATHS
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\newlightvibeexp\MANUAL_LINE_tracking.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

WAV = os.path.join(OUTDIR, "MANUAL_LINE_AUDIO.wav")
VIBRATION_PNG = os.path.join(OUTDIR, "MANUAL_LINE_VIBRATION.png")
SPECTRUM_PNG = os.path.join(OUTDIR, "MANUAL_LINE_SPECTRUM.png")
TRACKING_PNG = os.path.join(OUTDIR, "MANUAL_LINE_TRACKING_CHECK.png")

os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# IMPORTANT:
# Your actual marked line is here.
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
# TRACK THE GREEN MARKED LINE
# ============================================================

all_y = []
all_x = []

first_frame = None
first_mask = None

for frame_no in range(nframes):

    ret, frame = cap.read()

    if not ret:
        break

    if first_frame is None:
        first_frame = frame.copy()

    # ROI
    roi = frame[Y1:Y2, X1:X2]

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)

    # Green marker detection
    green = (
        (g > 100) &
        (g > r + 40) &
        (g > b + 40)
    )

    # Remove tiny noise
    ys, xs = np.where(green)

    if len(ys) == 0:
        all_y.append(np.nan)
        all_x.append(np.nan)
        continue

    # The actual marked line is the green component near
    # the centre of this ROI.
    #
    # Ignore isolated pixels.
    keep = (
        (ys > 20) &
        (ys < (Y2-Y1)-20)
    )

    ys = ys[keep]
    xs = xs[keep]

    if len(ys) == 0:
        all_y.append(np.nan)
        all_x.append(np.nan)
        continue

    # Median position of the marked line
    y = np.median(ys) + Y1
    x = np.median(xs) + X1

    all_y.append(y)
    all_x.append(x)

    if frame_no == 0:
        first_mask = green.copy()

cap.release()

all_y = np.asarray(all_y, dtype=np.float64)
all_x = np.asarray(all_x, dtype=np.float64)

# ============================================================
# CHECK
# ============================================================

valid = np.isfinite(all_y)

print()
print("Valid frames:", np.sum(valid), "/", len(all_y))
print("Y min:", np.nanmin(all_y))
print("Y max:", np.nanmax(all_y))
print("Y mean:", np.nanmean(all_y))
print("Y std:", np.nanstd(all_y))

if np.sum(valid) < 20:
    raise RuntimeError(
        "Could not detect the marked line. "
        "Check X1/X2/Y1/Y2."
    )

# Fill missing frames
idx = np.arange(len(all_y))

all_y = np.interp(
    idx,
    idx[valid],
    all_y[valid]
)

# ============================================================
# SAVE TRACKING CHECK
# ============================================================

check = first_frame.copy()

# Draw ROI
cv2.rectangle(
    check,
    (X1, Y1),
    (X2, Y2),
    (255, 0, 0),
    3
)

# Draw tracked positions
for x, y in zip(all_x, all_y):

    if np.isfinite(x) and np.isfinite(y):

        cv2.circle(
            check,
            (int(x), int(y)),
            3,
            (0, 0, 255),
            -1
        )

cv2.imwrite(TRACKING_PNG, check)

# ============================================================
# VERTICAL VIBRATION
# ============================================================

# Positive = line moved downward
# Negative = line moved upward

vibration = all_y - np.mean(all_y)

# DO NOT FILTER IT.
# This is the actual measured trajectory.

time = np.arange(len(vibration)) / fps

# ============================================================
# PLOT VIBRATION
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    time,
    vibration,
    linewidth=1
)

plt.xlabel("Time (seconds)")
plt.ylabel("Vertical displacement (pixels)")
plt.title("Measured Vertical Vibration of Marked Line")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    VIBRATION_PNG,
    dpi=200
)

plt.close()

# ============================================================
# CONSTRUCT AUDIO
#
# The tracked vibration is sampled at the camera FPS.
# We resample the measured waveform to 48 kHz ONLY
# so that it can be stored/listened to as WAV.
# ============================================================

duration = len(vibration) / fps

audio_time = np.linspace(
    0,
    duration,
    int(duration * 48000),
    endpoint=False
)

audio = np.interp(
    audio_time,
    time,
    vibration
)

# Remove DC
audio -= np.mean(audio)

# Normalize
peak = np.max(np.abs(audio))

if peak > 0:
    audio = audio / peak

# Safe listening level
audio *= 0.95

# ============================================================
# WRITE WAV
# ============================================================

write(
    WAV,
    48000,
    np.int16(audio * 32767)
)

print()
print("WAV written:")
print(WAV)

print("Audio duration:", len(audio) / 48000)
print("Audio RMS:", np.sqrt(np.mean(audio ** 2)))

# ============================================================
# SPECTRUM OF ACTUAL TRACKED VIBRATION
# ============================================================

f, p = welch(
    vibration,
    fs=fps,
    nperseg=min(256, len(vibration))
)

plt.figure(figsize=(14, 6))

plt.plot(f, p)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Spectrum of Measured Line Vibration")

plt.xlim(0, fps / 2)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    SPECTRUM_PNG,
    dpi=200
)

plt.close()

# ============================================================
# PRINT STRONGEST VIBRATION FREQUENCIES
# ============================================================

order = np.argsort(p)[::-1]

print()
print("STRONGEST MEASURED VIBRATION FREQUENCIES:")

shown = 0

for i in order:

    if f[i] < 0.1:
        continue

    print(
        f"{f[i]:8.3f} Hz   {p[i]:.8f}"
    )

    shown += 1

    if shown >= 15:
        break

print()
print("DONE")