import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, coherence

OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")

VISUAL_FILE = OUT / "straight_edge_tracking.npz"
ACTUAL_FILE = Path(r"C:\Users\Harkiran\Downloads\light vibe exp\actual_bass_audio.wav")

# ============================================================
# LOAD VISUAL STRAIGHT-EDGE DATA
# ============================================================

data = np.load(VISUAL_FILE)

deformation = data["deformation"].astype(np.float64)
fps = float(data["fps"])

print("Visual edge field:", deformation.shape)
print("Video FPS:", fps)

# Collapse spatial edge deformation into one robust signal
visual = np.median(
    deformation,
    axis=1
)

visual -= np.mean(visual)

# ============================================================
# LOAD CLEAN REFERENCE BASS
# ============================================================

fs_actual, actual = wavfile.read(ACTUAL_FILE)

actual = actual.astype(np.float64)

if actual.ndim > 1:
    actual = np.mean(actual, axis=1)

print("Reference audio:", len(actual), "samples")
print("Reference FS:", fs_actual)

# ============================================================
# RESAMPLE VISUAL SIGNAL TO AUDIO RATE
# ============================================================

t_visual = np.arange(len(visual)) / fps

duration = t_visual[-1]

t_audio = np.arange(
    0,
    duration,
    1.0 / fs_actual
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

    return sosfiltfilt(sos, x)

visual_bass = bass_filter(
    visual_48k,
    fs_actual
)

actual_bass = bass_filter(
    actual,
    fs_actual
)

# Match lengths
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
    nperseg=min(16384, N)
)

mask = (
    (f >= 20) &
    (f <= 250)
)

bf = f[mask]
bc = Cxy[mask]

idx = np.argmax(bc)

max_coherence = bc[idx]
max_coherence_freq = bf[idx]

# ============================================================
# VISUAL SPECTRUM
# ============================================================

fft = np.fft.rfft(
    visual_bass
)

freqs = np.fft.rfftfreq(
    N,
    1.0 / fs_actual
)

power = np.abs(fft)

mask = (
    (freqs >= 20) &
    (freqs <= 250)
)

vf = freqs[mask]
vp = power[mask]

top = np.argsort(vp)[-15:][::-1]

# ============================================================
# RESULTS
# ============================================================

print()
print("==============================================")
print("STRAIGHT EDGE vs CLEAN BASS")
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
print("Top visual frequencies:")
print()

for i in top:
    print(
        f"{vf[i]:8.2f} Hz    power={vp[i]:.5f}"
    )

print()
print("==============================================")

# ============================================================
# SAVE
# ============================================================

np.savez(
    OUT / "straight_edge_clean_bass_comparison.npz",
    visual_bass=visual_bass,
    actual_bass=actual_bass,
    visual_envelope=visual_env,
    actual_envelope=actual_env,
    frequencies=vf,
    visual_power=vp,
    coherence_frequencies=bf,
    coherence=bc,
    envelope_correlation=corr,
    max_coherence=max_coherence,
    max_coherence_frequency=max_coherence_freq
)

print()
print("Saved:")
print(
    OUT /
    "straight_edge_clean_bass_comparison.npz"
)
