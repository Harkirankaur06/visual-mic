# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch
from scipy.io.wavfile import write

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

INPUT = os.path.join(BASE, "marked_line_tracking.npz")

RAW_PLOT = os.path.join(BASE, "acoustic_waveform_FIXED.png")
RS_PLOT = os.path.join(BASE, "acoustic_waveform_RS_FIXED.png")
SPEC_PLOT = os.path.join(BASE, "acoustic_waveform_spectrum_FIXED.png")
WAV_OUT = os.path.join(BASE, "light_constructed_audio_FIXED.wav")

# ============================================================
# PARAMETERS
# ============================================================

FPS = 30.00075632158794
ROW_DELAY = 7.7435e-6
AUDIO_FS = 48000

LOWCUT = 20.0
HIGHCUT = 250.0

# ============================================================
# LOAD
# ============================================================

data = np.load(INPUT, allow_pickle=True)

tracked_y = np.asarray(data["tracked_y"], dtype=np.float64)

print("Input:", INPUT)
print("tracked_y:", tracked_y.shape)

n_frames, n_points = tracked_y.shape

# ============================================================
# CHECK INVALID DATA
# ============================================================

print()
print("INVALID DATA CHECK")

print("NaN count:", np.isnan(tracked_y).sum())
print("Inf count:", np.isinf(tracked_y).sum())

# Replace invalid values column-by-column.
for p in range(n_points):

    column = tracked_y[:, p]

    good = np.isfinite(column)

    if np.sum(good) >= 2:
        tracked_y[:, p] = np.interp(
            np.arange(n_frames),
            np.flatnonzero(good),
            column[good]
        )
    else:
        tracked_y[:, p] = 0.0

print("After cleanup:")
print("NaN count:", np.isnan(tracked_y).sum())
print("Inf count:", np.isinf(tracked_y).sum())

# ============================================================
# 1. FRAME-TO-FRAME VERTICAL MOTION
# ============================================================

# Difference between consecutive frames.
velocity = np.diff(tracked_y, axis=0)

# Remove per-point DC.
velocity -= np.mean(velocity, axis=0, keepdims=True)

# Common vertical motion.
raw_signal = np.median(velocity, axis=1)

# Remove remaining DC.
raw_signal -= np.mean(raw_signal)

t = np.arange(len(raw_signal)) / FPS

print()
print("FRAME MOTION")
print("min:", np.min(raw_signal))
print("max:", np.max(raw_signal))
print("RMS:", np.sqrt(np.mean(raw_signal ** 2)))

# ============================================================
# PLOT FRAME-RATE MOTION
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(t, raw_signal)

plt.xlabel("Time (s)")
plt.ylabel("Frame-to-frame vertical motion (pixels)")
plt.title("Frame-to-frame vertical motion of reflected-light line")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(RAW_PLOT, dpi=200)
plt.close()

# ============================================================
# 2. ROLLING-SHUTTER SAMPLES
# ============================================================

times = []
values = []

for f in range(1, n_frames):

    frame_time = f / FPS

    for p in range(n_points):

        y = tracked_y[f, p]

        value = velocity[f - 1, p]

        if not np.isfinite(y):
            continue

        if not np.isfinite(value):
            continue

        # Rolling-shutter capture time.
        capture_time = frame_time + y * ROW_DELAY

        times.append(capture_time)
        values.append(value)

times = np.asarray(times, dtype=np.float64)
values = np.asarray(values, dtype=np.float64)

# ============================================================
# SORT
# ============================================================

order = np.argsort(times)

times = times[order]
values = values[order]

# Remove any invalid samples.
good = np.isfinite(times) & np.isfinite(values)

times = times[good]
values = values[good]

print()
print("RS SAMPLES")
print("valid:", len(values))
print("time:", times[0], "to", times[-1])

# ============================================================
# REMOVE DUPLICATE TIMES
# ============================================================

unique_times, inverse = np.unique(times, return_inverse=True)

unique_values = np.zeros(len(unique_times))

counts = np.zeros(len(unique_times))

np.add.at(unique_values, inverse, values)
np.add.at(counts, inverse, 1)

unique_values /= np.maximum(counts, 1)

times = unique_times
values = unique_values

# ============================================================
# INTERPOLATE
# ============================================================

audio_time = np.arange(
    times[0],
    times[-1],
    1.0 / AUDIO_FS
)

rs_signal = np.interp(
    audio_time,
    times,
    values
)

# Safety cleanup.
rs_signal = np.nan_to_num(
    rs_signal,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# Remove mean.
rs_signal -= np.mean(rs_signal)

print("RS samples:", len(rs_signal))
print("RS duration:", len(rs_signal) / AUDIO_FS)
print("RS RMS:", np.sqrt(np.mean(rs_signal ** 2)))

# ============================================================
# 3. BANDPASS
# ============================================================

nyquist = AUDIO_FS / 2

b, a = butter(
    4,
    [LOWCUT / nyquist, HIGHCUT / nyquist],
    btype="band"
)

filtered = filtfilt(b, a, rs_signal)

filtered = np.nan_to_num(
    filtered,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# ============================================================
# 4. NORMALIZE
# ============================================================

peak = np.max(np.abs(filtered))

print("Filtered peak:", peak)

if peak > 1e-12:
    audio = filtered / peak
else:
    audio = filtered

audio = np.clip(audio * 0.9, -1, 1)

# ============================================================
# SAVE WAV
# ============================================================

write(
    WAV_OUT,
    AUDIO_FS,
    np.int16(audio * 32767)
)

print()
print("WAV:", WAV_OUT)

# ============================================================
# 5. PLOT RS WAVEFORM
# ============================================================

plot_seconds = min(7.0, len(audio) / AUDIO_FS)

N = int(plot_seconds * AUDIO_FS)

plt.figure(figsize=(14, 5))

plt.plot(
    np.arange(N) / AUDIO_FS,
    audio[:N],
    linewidth=0.5
)

plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")
plt.title("Rolling-shutter reconstructed optical-audio waveform")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(RS_PLOT, dpi=200)
plt.close()

# ============================================================
# 6. SPECTRUM
# ============================================================

freq, power = welch(
    audio,
    fs=AUDIO_FS,
    nperseg=min(65536, len(audio))
)

mask = (
    np.isfinite(freq)
    & np.isfinite(power)
    & (freq >= 1)
    & (freq <= 1000)
)

plt.figure(figsize=(14, 5))

plt.plot(
    freq[mask],
    power[mask]
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("Spectrum of reconstructed optical-audio signal")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(SPEC_PLOT, dpi=200)
plt.close()

# ============================================================
# 7. TOP FREQUENCIES
# ============================================================

idx = np.argsort(power[mask])[-20:][::-1]

print()
print("TOP FREQUENCIES")

for i in idx[:15]:

    f = freq[mask][i]
    p = power[mask][i]

    print(f"{f:8.2f} Hz   {p:.8e}")

print()
print("DONE")