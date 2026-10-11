import cv2
import numpy as np
from scipy.signal import detrend, welch
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import os

# ============================================================
# PATHS
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\newlightvibeexp\MANUAL_LINE_tracking.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

WAV_OUT = os.path.join(OUTDIR, "MANUAL_LINE_vertical_wave.wav")
WAVE_PNG = os.path.join(OUTDIR, "MANUAL_LINE_waveform.png")
SPEC_PNG = os.path.join(OUTDIR, "MANUAL_LINE_spectrum.png")

# ============================================================
# SETTINGS
# ============================================================

TARGET_FPS = 48000

# Rolling shutter delay measured for your A15
ROW_DELAY = 7.7435e-6

# Green tracking color
# We use green pixels from the manually marked line.
GREEN_THRESHOLD = 80

os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open MANUAL_LINE_tracking.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Resolution:", width, height)
print("Frames:", nframes)

# ============================================================
# EXTRACT GREEN LINE POSITION
# ============================================================

frame_y = []
frame_strength = []

for i in range(nframes):

    ret, frame = cap.read()

    if not ret:
        break

    # OpenCV BGR
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)

    # Green marker:
    # green must be stronger than red and blue
    green_score = g - ((r + b) // 2)

    mask = green_score > GREEN_THRESHOLD

    ys, xs = np.where(mask)

    if len(ys) > 20:

        # Median Y of all green pixels
        y = float(np.median(ys))

        strength = float(np.mean(green_score[mask]))

    else:

        y = np.nan
        strength = 0

    frame_y.append(y)
    frame_strength.append(strength)

cap.release()

frame_y = np.asarray(frame_y, dtype=np.float64)
frame_strength = np.asarray(frame_strength)

print("Tracked frames:", len(frame_y))
print("Valid Y:", np.sum(np.isfinite(frame_y)))

# ============================================================
# INTERPOLATE MISSING FRAMES
# ============================================================

valid = np.isfinite(frame_y)

if np.sum(valid) < 10:
    raise RuntimeError("Could not detect the green manual line.")

indices = np.arange(len(frame_y))

frame_y = np.interp(
    indices,
    indices[valid],
    frame_y[valid]
)

print("Y range:", frame_y.min(), frame_y.max())

# ============================================================
# REMOVE STATIC POSITION
# ============================================================

motion = frame_y - np.median(frame_y)

# Remove slow drift only.
# DO NOT high-pass at 20 Hz.
#
# The purpose here is to see what the line actually contains.

motion = detrend(motion, type="linear")

print("Motion RMS:", np.sqrt(np.mean(motion ** 2)))
print("Motion peak:", np.max(np.abs(motion)))

# ============================================================
# FRAME-TIME SIGNAL
# ============================================================

frame_time = np.arange(len(motion)) / fps

# ============================================================
# BUILD ROLLING-SHUTTER WAVE
#
# The line position gives us a row coordinate.
#
# Each frame therefore gets a slightly different timestamp
# according to the row position.
# ============================================================

row_time = frame_time + frame_y * ROW_DELAY

# Sort chronologically
order = np.argsort(row_time)

row_time = row_time[order]
motion_rs = motion[order]

# Remove duplicate timestamps if any
unique_time, unique_idx = np.unique(row_time, return_index=True)

row_time = unique_time
motion_rs = motion_rs[unique_idx]

print("Rolling-shutter samples:", len(motion_rs))
print("Time range:", row_time[0], row_time[-1])

# ============================================================
# IMPORTANT:
# A SINGLE LINE POSITION PER FRAME IS ONLY ~30 Hz.
#
# So first create the DIRECT 30-FPS WAVE.
# Then also create the rolling-shutter interpolation.
# ============================================================

# ------------------------------------------------------------
# DIRECT FRAME WAVE
# ------------------------------------------------------------

direct_time = np.arange(len(motion)) / fps

direct_audio = np.interp(
    np.linspace(
        direct_time[0],
        direct_time[-1],
        int((direct_time[-1] - direct_time[0]) * TARGET_FPS)
    ),
    direct_time,
    motion
)

# ------------------------------------------------------------
# ROLLING-SHUTTER INTERPOLATED WAVE
# ------------------------------------------------------------

duration = row_time[-1] - row_time[0]

N = int(duration * TARGET_FPS)

high_time = np.linspace(
    row_time[0],
    row_time[-1],
    N
)

audio_rs = np.interp(
    high_time,
    row_time,
    motion_rs
)

# ============================================================
# NORMALIZE
# ============================================================

def normalize_audio(x):

    x = x.astype(np.float64)

    x -= np.mean(x)

    peak = np.max(np.abs(x))

    if peak < 1e-12:
        return x

    return x / peak


audio_rs = normalize_audio(audio_rs)

# Add reasonable listening gain.
# This is ONLY amplitude scaling.
audio_rs *= 0.95

# ============================================================
# SAVE WAV
# ============================================================

write(
    WAV_OUT,
    TARGET_FPS,
    np.int16(np.clip(audio_rs, -1, 1) * 32767)
)

print()
print("WAV:", WAV_OUT)
print("Duration:", len(audio_rs) / TARGET_FPS, "seconds")
print("RMS:", np.sqrt(np.mean(audio_rs ** 2)))
print("Peak:", np.max(np.abs(audio_rs)))

# ============================================================
# WAVEFORM
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(high_time, audio_rs)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Manual Line - Rolling Shutter Reconstructed Wave")

plt.tight_layout()
plt.savefig(WAVE_PNG, dpi=150)
plt.close()

# ============================================================
# RAW SPECTRUM
# ============================================================

# Do NOT filter or notch anything here.
# We want to see what is ACTUALLY present.

nperseg = min(65536, len(audio_rs))

freq, power = welch(
    audio_rs,
    fs=TARGET_FPS,
    nperseg=nperseg
)

# Limit plot to 2000 Hz
mask = freq <= 2000

plt.figure(figsize=(14, 5))

plt.plot(freq[mask], power[mask])

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Manual Line - RAW Rolling Shutter Spectrum")

plt.xlim(0, 2000)

plt.tight_layout()
plt.savefig(SPEC_PNG, dpi=150)
plt.close()

# ============================================================
# PRINT STRONGEST FREQUENCIES
# ============================================================

idx = np.argsort(power[mask])[-30:][::-1]

print()
print("TOP RAW FREQUENCIES:")

for j in idx[:20]:

    print(
        f"{freq[mask][j]:9.3f} Hz   "
        f"{power[mask][j]:.8f}"
    )

print()
print("DONE")