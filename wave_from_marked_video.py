# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.io.wavfile import write

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

VIDEO = os.path.join(BASE, "marked_line_tracking.mp4")

RAW_PLOT = os.path.join(BASE, "wave_from_marked_video.png")
FILTERED_PLOT = os.path.join(BASE, "wave_from_marked_video_filtered.png")
SPECTRUM_PLOT = os.path.join(BASE, "wave_from_marked_video_spectrum.png")
WAV_OUT = os.path.join(BASE, "wave_from_marked_video.wav")

# ============================================================
# CAMERA PARAMETERS
# ============================================================

ROW_DELAY = 7.7435e-6
AUDIO_FS = 48000

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open marked_line_tracking.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video:", VIDEO)
print("FPS:", fps)
print("Frames:", frames)

# ============================================================
# GREEN MARK DETECTION
# ============================================================

ys = []
times = []

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green overlay detection.
    #
    # OpenCV HSV:
    # green is approximately H=35-85
    #
    mask = cv2.inRange(
        hsv,
        np.array([35, 60, 60]),
        np.array([90, 255, 255])
    )

    # Remove tiny isolated pixels.
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find green pixels.
    yy, xx = np.where(mask > 0)

    if len(yy) > 20:

        # We want the tracked line region.
        #
        # Ignore extreme image borders.
        valid = (
            (yy > 700) &
            (yy < 950) &
            (xx > 300) &
            (xx < 1450)
        )

        yy2 = yy[valid]

        if len(yy2) > 10:

            # Median Y position of the marked line.
            y_position = np.median(yy2)

            ys.append(y_position)
            times.append(frame_number / fps)

    frame_number += 1

cap.release()

ys = np.asarray(ys, dtype=np.float64)
times = np.asarray(times, dtype=np.float64)

print()
print("Detected marked-line frames:", len(ys))

if len(ys) < 20:
    raise RuntimeError(
        "Not enough green pixels detected. "
        "The overlay may use a different color."
    )

print("Y range:", np.min(ys), np.max(ys))

# ============================================================
# INTERPOLATE MISSING FRAMES
# ============================================================

full_times = np.arange(frames) / fps

full_y = np.interp(
    full_times,
    times,
    ys
)

# ============================================================
# RAW Y WAVEFORM
# ============================================================

# Remove constant offset.
raw = full_y - np.mean(full_y)

plt.figure(figsize=(14, 5))

plt.plot(
    full_times,
    raw,
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("Marked-line Y displacement (pixels)")
plt.title("Vertical motion extracted directly from marked tracking video")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(RAW_PLOT, dpi=200)
plt.close()

print("Saved:", RAW_PLOT)

# ============================================================
# REMOVE SLOW DRIFT
# ============================================================

# Smooth low-frequency movement.
slow = cv2.GaussianBlur(
    raw.astype(np.float32).reshape(-1, 1),
    (0, 0),
    sigmaX=15
).flatten()

motion = raw - slow

# ============================================================
# PLOT MOTION
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    full_times,
    motion,
    linewidth=0.8
)

plt.xlabel("Time (s)")
plt.ylabel("Fast vertical displacement (pixels)")
plt.title("High-frequency component of marked-line motion")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(FILTERED_PLOT, dpi=200)
plt.close()

print("Saved:", FILTERED_PLOT)

# ============================================================
# ROLLING-SHUTTER RECONSTRUCTION
# ============================================================

rs_times = []
rs_values = []

for i in range(len(full_y)):

    y = full_y[i]

    value = motion[i]

    capture_time = (
        i / fps
        + y * ROW_DELAY
    )

    rs_times.append(capture_time)
    rs_values.append(value)

rs_times = np.asarray(rs_times)
rs_values = np.asarray(rs_values)

# Sort.
order = np.argsort(rs_times)

rs_times = rs_times[order]
rs_values = rs_values[order]

# Remove invalid values.
good = (
    np.isfinite(rs_times)
    & np.isfinite(rs_values)
)

rs_times = rs_times[good]
rs_values = rs_values[good]

# ============================================================
# INTERPOLATE TO AUDIO RATE
# ============================================================

audio_time = np.arange(
    rs_times[0],
    rs_times[-1],
    1.0 / AUDIO_FS
)

audio = np.interp(
    audio_time,
    rs_times,
    rs_values
)

audio = np.nan_to_num(audio)

audio -= np.mean(audio)

# ============================================================
# AUDIO BANDPASS
# ============================================================

nyquist = AUDIO_FS / 2

b, a = butter(
    4,
    [20 / nyquist, 250 / nyquist],
    btype="band"
)

audio = filtfilt(b, a, audio)

audio = np.nan_to_num(audio)

# Normalize.
peak = np.max(np.abs(audio))

print()
print("Reconstructed RMS:", np.sqrt(np.mean(audio ** 2)))
print("Reconstructed peak:", peak)

if peak > 1e-12:
    audio = audio / peak

audio *= 0.9

# ============================================================
# SAVE WAV
# ============================================================

write(
    WAV_OUT,
    AUDIO_FS,
    np.int16(np.clip(audio, -1, 1) * 32767)
)

print("Saved:", WAV_OUT)

# ============================================================
# SPECTRUM
# ============================================================

freq, power = welch(
    audio,
    fs=AUDIO_FS,
    nperseg=min(65536, len(audio))
)

mask = (
    (freq >= 1)
    & (freq <= 1000)
    & np.isfinite(power)
)

plt.figure(figsize=(14, 5))

plt.plot(
    freq[mask],
    power[mask]
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Spectrum of audio reconstructed from marked video")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(SPECTRUM_PLOT, dpi=200)
plt.close()

print("Saved:", SPECTRUM_PLOT)

# ============================================================
# TOP FREQUENCIES
# ============================================================

idx = np.argsort(power[mask])[-20:][::-1]

print()
print("TOP FREQUENCIES")

for i in idx[:15]:

    print(
        f"{freq[mask][i]:8.2f} Hz   "
        f"{power[mask][i]:.8e}"
    )

print()
print("DONE")