import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.io import wavfile
from pathlib import Path

# ============================================================
# INPUT / OUTPUT
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUTDIR = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOCKED TARGET ROI
# ============================================================

X1, X2 = 350, 1500
Y1, Y2 = 740, 880

# Samsung A15 rolling-shutter calibration
ROW_DELAY = 7.7435e-6

# Audio output
FS = 48000

# ============================================================
# VIDEO INFO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video:", width, "x", height)
print("FPS:", fps)
print("Frames:", nframes)
print("Duration:", nframes / fps)

# ============================================================
# TRACK BRIGHT REFLECTION LINE
# ============================================================

tracks = []

# Expected position of the white line inside ROI.
# Start around the middle of the confirmed vertical range.
prev_y = (Y1 + Y2) // 2

frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[Y1:Y2, X1:X2].astype(np.float32)

    # Slight vertical smoothing:
    # suppress pixel noise while retaining the reflection ridge.
    smooth = cv2.GaussianBlur(roi, (1, 7), 0)

    # --------------------------------------------------------
    # For each x-column, find the brightest ridge near previous
    # position. This prevents jumping to unrelated reflections.
    # --------------------------------------------------------

    current_y = []

    prev_local = prev_y - Y1

    for x in range(smooth.shape[1]):

        # Search only within ±25 pixels of previous position.
        lo = max(0, int(prev_local - 25))
        hi = min(smooth.shape[0], int(prev_local + 26))

        column = smooth[lo:hi, x]

        if len(column) == 0:
            current_y.append(prev_local)
            continue

        # Brightest point
        peak = np.argmax(column)
        candidate = lo + peak

        # Local weighted centroid around peak
        a = max(0, candidate - 3)
        b = min(smooth.shape[0], candidate + 4)

        weights = smooth[a:b, x] + 1e-6
        yy = np.arange(a, b)

        weighted = np.sum(yy * weights) / np.sum(weights)

        current_y.append(weighted)

    current_y = np.asarray(current_y)

    # Robustly smooth the tracked line horizontally.
    current_y = cv2.GaussianBlur(
        current_y.reshape(1, -1).astype(np.float32),
        (31, 1),
        0
    ).ravel()

    # Median position gives one displacement value per frame.
    median_y = np.median(current_y)

    # Keep continuity.
    prev_y = 0.85 * prev_y + 0.15 * (median_y + Y1)

    tracks.append(current_y + Y1)

    frame_idx += 1

    if frame_idx % 25 == 0:
        print("Processed:", frame_idx, "/", nframes)

cap.release()

tracks = np.asarray(tracks)

print("\nTracking matrix:", tracks.shape)

# ============================================================
# REMOVE STATIC LINE SHAPE
# ============================================================

# The absolute line position differs by column.
# We care about changes caused by the ripples.
reference = np.median(tracks, axis=0)

displacement = tracks - reference[None, :]

# Remove frame-wide vertical drift.
displacement -= np.median(displacement, axis=1)[:, None]

# ============================================================
# SAVE TRACKING DATA
# ============================================================

np.savez(
    OUTDIR / "light_reflection_tracking.npz",
    tracks=tracks,
    displacement=displacement,
    fps=fps,
    row_delay=ROW_DELAY,
    roi=np.array([X1, X2, Y1, Y2])
)

# ============================================================
# ROLLING-SHUTTER TEMPORAL RECONSTRUCTION
# ============================================================

# Each image row corresponds to a slightly different time.
#
# t = frame_time + row * ROW_DELAY
#
# Convert every tracked displacement sample into its
# corresponding rolling-shutter time.

times = []
values = []

for i in range(nframes):

    frame_time = i / fps

    for x in range(displacement.shape[1]):

        # Use displacement of the tracked reflection.
        value = displacement[i, x]

        # Track's vertical position determines the row time.
        row = tracks[i, x]

        t = frame_time + row * ROW_DELAY

        times.append(t)
        values.append(value)

times = np.asarray(times)
values = np.asarray(values)

# Sort chronologically.
order = np.argsort(times)

times = times[order]
values = values[order]

# ============================================================
# REMOVE DUPLICATES / INVALID VALUES
# ============================================================

valid = np.isfinite(times) & np.isfinite(values)

times = times[valid]
values = values[valid]

# ============================================================
# INTERPOLATE TO UNIFORM HIGH-RATE SIGNAL
# ============================================================

duration = times[-1] - times[0]

t_uniform = np.arange(
    times[0],
    times[-1],
    1 / FS
)

signal = np.interp(
    t_uniform,
    times,
    values
)

signal = detrend(signal)

# ============================================================
# BANDPASS
# ============================================================

# Bass experiment:
# Keep low/mid acoustic frequencies while removing DC and
# very slow water drift.

low = 20
high = min(8000, FS / 2 - 100)

sos = butter(
    5,
    [low, high],
    btype="bandpass",
    fs=FS,
    output="sos"
)

signal = sosfiltfilt(sos, signal)

# ============================================================
# NORMALIZE
# ============================================================

signal -= np.mean(signal)

peak = np.max(np.abs(signal))

if peak > 0:
    signal /= peak

# Conservative output level
signal *= 0.85

audio = np.int16(
    np.clip(signal, -1, 1) * 32767
)

# ============================================================
# WRITE WAV
# ============================================================

outfile = OUTDIR / "light_reflection_RS_recovered.wav"

wavfile.write(
    outfile,
    FS,
    audio
)

print("\nDONE")
print("Output:", outfile)
print("Samples:", len(audio))
print("Duration:", len(audio) / FS, "seconds")
