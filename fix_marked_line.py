# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

NPZ = os.path.join(
    BASE,
    "marked_line_tracking.npz"
)

OUT_WAV = os.path.join(
    BASE,
    "marked_line_RS_bass_FIXED.wav"
)

OUT_PLOT = os.path.join(
    BASE,
    "marked_line_vertical_motion_FIXED.png"
)

OUT_COMPARE = os.path.join(
    BASE,
    "marked_line_vs_actual_bass_FIXED.png"
)

REFERENCE = r"C:\Users\Harkiran\Downloads\light vibe exp\actual_bass_audio.wav"

# ============================================================
# LOAD EXISTING TRACK
# ============================================================

d = np.load(NPZ)

tracked_y = d["tracked_y"]
baseline = d["baseline"]
x = d["x"]

fps = float(d["fps"])
row_delay = float(d["row_delay"])

print("Tracked Y:", tracked_y.shape)
print("FPS:", fps)
print("Row delay:", row_delay)

nframes, nx = tracked_y.shape

t = np.arange(nframes) / fps

# ============================================================
# IMPORTANT:
# RECONSTRUCT THE REAL VERTICAL MOTION
#
# DO NOT subtract the frame median.
# The common vertical movement IS the signal.
# ============================================================

motion = tracked_y - baseline[None, :]

# ============================================================
# LINE-WISE MOTION SIGNALS
# ============================================================

median_motion = np.median(
    motion,
    axis=1
)

mean_motion = np.mean(
    motion,
    axis=1
)

# Spatial RMS is also useful
rms_motion = np.sqrt(
    np.mean(
        motion ** 2,
        axis=1
    )
)

print()
print("MOTION")

for name, s in [
    ("median", median_motion),
    ("mean", mean_motion),
    ("rms", rms_motion)
]:

    print(
        name,
        "RMS:",
        np.sqrt(np.mean(s ** 2)),
        "min:",
        np.min(s),
        "max:",
        np.max(s)
    )

# ============================================================
# PLOT REAL LINE MOTION
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    t,
    median_motion,
    linewidth=2,
    label="Median line motion"
)

plt.plot(
    t,
    mean_motion,
    linewidth=1,
    alpha=0.7,
    label="Mean line motion"
)

plt.xlabel("Time (s)")
plt.ylabel("Vertical displacement (pixels)")
plt.title("Actual Tracked Vertical Motion of the Marked Line")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_PLOT,
    dpi=160
)

plt.close()

# ============================================================
# SPECTRUM OF LINE MOTION
# ============================================================

s = median_motion.copy()

s -= np.mean(s)

window = np.hanning(len(s))

F = np.fft.rfft(
    s * window
)

freq = np.fft.rfftfreq(
    len(s),
    1 / fps
)

mag = np.abs(F)

plt.figure(figsize=(14, 5))

plt.plot(
    freq,
    mag
)

plt.xlim(
    0,
    fps / 2
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum of Marked-Line Vertical Motion")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE,
        "marked_line_vertical_spectrum_FIXED.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# PRINT STRONGEST COMPONENTS
# ============================================================

valid = freq > 0.1

fv = freq[valid]
mv = mag[valid]

peaks, _ = signal.find_peaks(
    mv
)

order = np.argsort(
    mv[peaks]
)[::-1]

print()
print("STRONGEST LINE-MOTION FREQUENCIES")

for k in order[:15]:

    p = peaks[k]

    print(
        f"{fv[p]:8.3f} Hz   "
        f"{mv[p]:.4f}"
    )

# ============================================================
# ROLLING-SHUTTER AUDIO
# ============================================================

print()
print("Building corrected rolling-shutter signal...")

sample_times = []
sample_values = []

for f in range(nframes):

    frame_time = f / fps

    ys = tracked_y[f]

    values = motion[f]

    # Each vertical position gets its own capture time
    times = (
        frame_time
        +
        ys * row_delay
    )

    sample_times.extend(
        times.tolist()
    )

    sample_values.extend(
        values.tolist()
    )

sample_times = np.asarray(
    sample_times,
    dtype=float
)

sample_values = np.asarray(
    sample_values,
    dtype=float
)

# Sort by time
order = np.argsort(
    sample_times
)

sample_times = sample_times[order]
sample_values = sample_values[order]

# Remove duplicate timestamps
dt = np.diff(
    sample_times
)

keep = np.concatenate(
    (
        [True],
        dt > 1e-10
    )
)

sample_times = sample_times[keep]
sample_values = sample_values[keep]

# ============================================================
# RESAMPLE
# ============================================================

SR = 48000

audio_t = np.arange(
    sample_times[0],
    sample_times[-1],
    1.0 / SR
)

audio = np.interp(
    audio_t,
    sample_times,
    sample_values
)

audio -= np.mean(audio)

# ============================================================
# BASS FILTER
# ============================================================

b, a = signal.butter(
    4,
    [
        20 / (SR / 2),
        250 / (SR / 2)
    ],
    btype="band"
)

audio = signal.filtfilt(
    b,
    a,
    audio
)

# Remove extreme spikes
sd = np.std(audio)

audio = np.clip(
    audio,
    -5 * sd,
    5 * sd
)

# Normalize
audio /= (
    np.max(
        np.abs(audio)
    )
    + 1e-12
)

audio *= 0.85

wavfile.write(
    OUT_WAV,
    SR,
    np.int16(
        audio * 32767
    )
)

print()
print("WAV written:")
print(OUT_WAV)

# ============================================================
# ACTUAL BASS ENVELOPE COMPARISON
# ============================================================

if os.path.exists(REFERENCE):

    ref_fs, ref = wavfile.read(
        REFERENCE
    )

    ref = np.asarray(
        ref,
        dtype=float
    )

    if ref.ndim > 1:
        ref = np.mean(
            ref,
            axis=1
        )

    ref -= np.mean(ref)

    # Actual bass envelope
    ref_env = np.abs(
        signal.hilbert(ref)
    )

    k = max(
        3,
        int(ref_fs * 0.05)
    )

    ref_env = np.convolve(
        ref_env,
        np.ones(k) / k,
        mode="same"
    )

    ref_t = np.arange(
        len(ref)
    ) / ref_fs

    ref_frame = np.interp(
        t,
        ref_t,
        ref_env
    )

    # Visual envelope
    visual_env = np.abs(
        signal.hilbert(
            median_motion -
            np.mean(median_motion)
        )
    )

    k2 = max(
        3,
        int(fps * 0.15)
    )

    visual_env = np.convolve(
        visual_env,
        np.ones(k2) / k2,
        mode="same"
    )

    # Normalize
    visual_env -= np.mean(
        visual_env
    )

    visual_env /= (
        np.std(visual_env)
        + 1e-12
    )

    ref_frame -= np.mean(
        ref_frame
    )

    ref_frame /= (
        np.std(ref_frame)
        + 1e-12
    )

    corr = np.corrcoef(
        visual_env,
        ref_frame
    )[0, 1]

    print()
    print("VISUAL VS ACTUAL BASS")
    print(
        "Envelope correlation:",
        corr
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        t,
        visual_env,
        label="Marked-line visual envelope",
        linewidth=2
    )

    plt.plot(
        t,
        ref_frame,
        label="Actual bass envelope",
        linewidth=1.5
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Normalized envelope")

    plt.title(
        "Marked-Line Motion vs Actual Bass"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT_COMPARE,
        dpi=160
    )

    plt.close()

# ============================================================
# DONE
# ============================================================

print()
print("========================================")
print("CORRECTED LINE ANALYSIS COMPLETE")
print("========================================")
print()
print("Motion plot:")
print(OUT_PLOT)
print()
print("Spectrum:")
print(
    os.path.join(
        BASE,
        "marked_line_vertical_spectrum_FIXED.png"
    )
)
print()
print("WAV:")
print(OUT_WAV)
print("========================================")