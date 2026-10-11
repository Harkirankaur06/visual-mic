# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

TRACK_FILE = r"C:\Users\Harkiran\Downloads\newlightvibeexp\horizontal_line_VERTICAL_tracking.npy"

REFERENCE_FILE = r"C:\Users\Harkiran\Downloads\light vibe exp\actual_bass_audio.wav"

OUT_DIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp\vertical_signal_analysis"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

a = np.load(TRACK_FILE)

frame = a[:, 0].astype(int)
point = a[:, 1].astype(int)
x = a[:, 2]
y = a[:, 3]
disp = a[:, 4]

fps = 30.00075632158794

frames = np.unique(frame)

print("Frames:", len(frames))
print("Points:", len(np.unique(point)))
print("Displacement min:", disp.min())
print("Displacement max:", disp.max())
print("Displacement std:", disp.std())

# ============================================================
# FRAME-WISE SIGNALS
# ============================================================

median_disp = []
mean_disp = []
std_disp = []
max_disp = []
min_disp = []

for f in frames:

    d = disp[frame == f]

    median_disp.append(np.median(d))
    mean_disp.append(np.mean(d))
    std_disp.append(np.std(d))
    max_disp.append(np.max(d))
    min_disp.append(np.min(d))

median_disp = np.asarray(median_disp)
mean_disp = np.asarray(mean_disp)
std_disp = np.asarray(std_disp)
max_disp = np.asarray(max_disp)
min_disp = np.asarray(min_disp)

t = np.arange(len(frames)) / fps

# ============================================================
# PRINT
# ============================================================

print("\nSIGNAL STATISTICS")

for name, s in [
    ("median", median_disp),
    ("mean", mean_disp),
    ("std", std_disp),
    ("max", max_disp),
    ("min", min_disp)
]:

    print(
        name,
        "RMS =",
        np.sqrt(np.mean(s ** 2)),
        "range =",
        np.min(s),
        np.max(s)
    )

# ============================================================
# PLOT FRAME-WISE DISPLACEMENT
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(t, median_disp, label="Median")
plt.plot(t, mean_disp, label="Mean", alpha=0.7)
plt.plot(t, std_disp, label="Spatial std", alpha=0.7)

plt.xlabel("Time (s)")
plt.ylabel("Tracked displacement")
plt.title("Frame-wise Vertical Motion")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "01_framewise_motion.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# TEST EACH SIGNAL
# ============================================================

signals = {
    "median": median_disp,
    "mean": mean_disp,
    "std": std_disp,
    "max": max_disp,
    "min": min_disp
}

# ============================================================
# SPECTRA
# ============================================================

plt.figure(figsize=(14, 6))

for name, s in signals.items():

    s = s - np.mean(s)

    F = np.fft.rfft(
        s * np.hanning(len(s))
    )

    freq = np.fft.rfftfreq(
        len(s),
        1 / fps
    )

    mag = np.abs(F)

    mag /= np.max(mag) + 1e-12

    plt.plot(
        freq,
        mag,
        label=name
    )

plt.xlim(0, 15)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized magnitude")
plt.title("Vertical Motion Spectra")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "02_motion_spectra.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# PRINT PEAKS
# ============================================================

print("\nSTRONGEST FREQUENCIES")

for name, s in signals.items():

    s = s - np.mean(s)

    F = np.fft.rfft(
        s * np.hanning(len(s))
    )

    freq = np.fft.rfftfreq(
        len(s),
        1 / fps
    )

    mag = np.abs(F)

    valid = freq > 0.1

    f2 = freq[valid]
    m2 = mag[valid]

    peaks, _ = signal.find_peaks(m2)

    order = np.argsort(
        m2[peaks]
    )[::-1]

    print("\n", name)

    for k in order[:8]:

        p = peaks[k]

        print(
            f"{f2[p]:7.3f} Hz   "
            f"{m2[p]:.4f}"
        )

# ============================================================
# SPECTROGRAM OF BEST SIMPLE SIGNAL
# ============================================================

best_signal = std_disp.copy()

best_signal -= np.mean(best_signal)

f, tt, S = signal.spectrogram(
    best_signal,
    fs=fps,
    nperseg=64,
    noverlap=32
)

plt.figure(figsize=(14, 6))

plt.pcolormesh(
    tt,
    f,
    10 * np.log10(S + 1e-12),
    shading="auto"
)

plt.ylim(0, 15)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Tracked Motion Spectrogram")

plt.colorbar(label="dB")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "03_motion_spectrogram.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# REFERENCE AUDIO
# ============================================================

ref_fs, ref = wavfile.read(
    REFERENCE_FILE
)

ref = np.asarray(ref, dtype=float)

if ref.ndim > 1:
    ref = np.mean(ref, axis=1)

ref -= np.mean(ref)

# Envelope
ref_env = np.abs(
    signal.hilbert(ref)
)

smooth = int(
    ref_fs * 0.05
)

kernel = np.ones(smooth) / smooth

ref_env = np.convolve(
    ref_env,
    kernel,
    mode="same"
)

ref_t = np.arange(
    len(ref)
) / ref_fs

ref_frame_env = np.interp(
    t,
    ref_t,
    ref_env
)

ref_frame_env -= np.mean(ref_frame_env)

ref_frame_env /= (
    np.std(ref_frame_env)
    + 1e-12
)

# ============================================================
# COMPARE ALL VISUAL SIGNALS
# ============================================================

print("\nENVELOPE CORRELATIONS")

for name, s in signals.items():

    s = np.asarray(s)

    # Use absolute displacement for envelope
    env = np.abs(
        signal.hilbert(
            s - np.mean(s)
        )
    )

    # Smooth
    k = max(
        3,
        int(fps * 0.15)
    )

    env = np.convolve(
        env,
        np.ones(k) / k,
        mode="same"
    )

    env -= np.mean(env)

    env /= (
        np.std(env)
        + 1e-12
    )

    corr = np.corrcoef(
        env,
        ref_frame_env
    )[0, 1]

    print(
        f"{name:8s}: {corr:.5f}"
    )

# ============================================================
# PLOT BEST COMPARISON
# ============================================================

visual = np.abs(
    signal.hilbert(
        std_disp - np.mean(std_disp)
    )
)

k = max(
    3,
    int(fps * 0.15)
)

visual = np.convolve(
    visual,
    np.ones(k) / k,
    mode="same"
)

visual -= np.mean(visual)

visual /= (
    np.std(visual)
    + 1e-12
)

plt.figure(figsize=(14, 6))

plt.plot(
    t,
    visual,
    label="Visual motion envelope",
    linewidth=1.5
)

plt.plot(
    t,
    ref_frame_env,
    label="Actual bass envelope",
    linewidth=1.5
)

plt.xlabel("Time (s)")
plt.ylabel("Normalized envelope")

plt.title(
    "Visual Motion vs Actual Bass"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "04_visual_vs_actual_bass.png"
    ),
    dpi=160
)

plt.close()

print("\n================================")
print("DONE")
print("================================")
print(OUT_DIR)