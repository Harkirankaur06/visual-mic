import cv2
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, coherence
import subprocess
import imageio_ffmpeg

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"
OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")

# ============================================================
# LOAD STRAIGHT-EDGE TRACKING
# ============================================================

data = np.load(
    OUT / "straight_edge_tracking.npz"
)

profiles = data["profiles"].astype(np.float64)
deformation = data["deformation"].astype(np.float64)
fps = float(data["fps"])

print("Visual edge:", deformation.shape)
print("Video FPS:", fps)

# ============================================================
# MAKE ONE VISUAL SIGNAL FROM EDGE DEFORMATION
# ============================================================

# Robust spatial average of the moving edge.
visual = np.median(
    deformation,
    axis=1
)

visual -= np.mean(visual)

# ============================================================
# EXTRACT ORIGINAL AUDIO FROM VIDEO
# ============================================================

actual_wav = OUT / "light_actual_audio.wav"

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

cmd = [
    ffmpeg,
    "-y",
    "-i", VIDEO,
    "-vn",
    "-ac", "1",
    "-ar", "48000",
    str(actual_wav)
]

subprocess.run(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

fs_actual, actual = wavfile.read(actual_wav)

actual = actual.astype(np.float64)

# ============================================================
# RESAMPLE VISUAL SIGNAL TO 48 kHz
# ============================================================

t_visual = np.arange(
    len(visual)
) / fps

duration = t_visual[-1]

t_audio = np.arange(
    0,
    duration,
    1 / fs_actual
)

visual_48k = np.interp(
    t_audio,
    t_visual,
    visual
)

# ============================================================
# BASS FILTER
# ============================================================

def bass_filter(x, fs):

    sos = butter(
        5,
        [20, 250],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        x
    )

visual_bass = bass_filter(
    visual_48k,
    fs_actual
)

actual_bass = bass_filter(
    actual[:len(visual_bass)],
    fs_actual
)

N = min(
    len(visual_bass),
    len(actual_bass)
)

visual_bass = visual_bass[:N]
actual_bass = actual_bass[:N]

# ============================================================
# NORMALIZE
# ============================================================

visual_bass /= (
    np.std(visual_bass) + 1e-12
)

actual_bass /= (
    np.std(actual_bass) + 1e-12
)

# ============================================================
# ENVELOPE CORRELATION
# ============================================================

window = int(
    0.05 * fs_actual
)

kernel = np.ones(window) / window

visual_env = np.sqrt(
    np.convolve(
        visual_bass ** 2,
        kernel,
        mode="same"
    )
)

actual_env = np.sqrt(
    np.convolve(
        actual_bass ** 2,
        kernel,
        mode="same"
    )
)

corr = np.corrcoef(
    visual_env,
    actual_env
)[0, 1]

# ============================================================
# COHERENCE
# ============================================================

f, Cxy = coherence(
    actual_bass,
    visual_bass,
    fs=fs_actual,
    nperseg=min(
        16384,
        N
    )
)

bass_mask = (
    (f >= 20) &
    (f <= 250)
)

bass_f = f[bass_mask]
bass_c = Cxy[bass_mask]

max_idx = np.argmax(bass_c)

max_coherence = bass_c[max_idx]
max_coherence_freq = bass_f[max_idx]

# ============================================================
# VISUAL SPECTRUM
# ============================================================

fft = np.fft.rfft(
    visual_bass
)

freqs = np.fft.rfftfreq(
    N,
    1 / fs_actual
)

power = np.abs(fft)

mask = (
    (freqs >= 20) &
    (freqs <= 250)
)

bf = freqs[mask]
bp = power[mask]

top = np.argsort(bp)[-10:][::-1]

# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("==============================================")
print("STRAIGHT EDGE vs ACTUAL BASS")
print("==============================================")
print()
print("Envelope correlation:")
print(corr)
print()
print("Maximum coherence:")
print(max_coherence)
print()
print("Frequency of maximum coherence:")
print(max_coherence_freq, "Hz")
print()
print("Top visual frequencies in 20-250 Hz:")

for i in top:
    print(
        f"{bf[i]:8.2f} Hz   power={bp[i]:.4f}"
    )

print()
print("==============================================")

# ============================================================
# SAVE RESULTS
# ============================================================

np.savez(
    OUT / "straight_edge_bass_comparison.npz",
    visual_bass=visual_bass,
    actual_bass=actual_bass,
    visual_envelope=visual_env,
    actual_envelope=actual_env,
    frequencies=bf,
    visual_power=bp,
    coherence_frequencies=bass_f,
    coherence=bass_c,
    envelope_correlation=corr,
    max_coherence=max_coherence,
    max_coherence_frequency=max_coherence_freq
)

print()
print("Results saved to:")
print(
    OUT /
    "straight_edge_bass_comparison.npz"
)
