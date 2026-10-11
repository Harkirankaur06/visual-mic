import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import imageio_ffmpeg

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\light vibe exp"

os.makedirs(OUTDIR, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

actual_wav = os.path.join(OUTDIR, "actual_bass_audio.wav")

# ---------------------------------------------------------
# Extract original audio from video
# ---------------------------------------------------------

cmd = [
    FFMPEG,
    "-y",
    "-i", VIDEO,
    "-vn",
    "-ac", "1",
    "-ar", "48000",
    "-c:a", "pcm_s16le",
    actual_wav
]

subprocess.run(cmd, check=True)

print("Actual audio saved:", actual_wav)

# ---------------------------------------------------------
# Load actual + reconstructed audio
# ---------------------------------------------------------

fs_actual, actual = wavfile.read(actual_wav)
actual = actual.astype(np.float64)

recon_path = os.path.join(
    OUTDIR,
    "reflection_rs_wide.wav"
)

fs_recon, recon = wavfile.read(recon_path)
recon = recon.astype(np.float64)

# Normalize
actual /= max(np.max(np.abs(actual)), 1)
recon /= max(np.max(np.abs(recon)), 1)

# ---------------------------------------------------------
# Spectrogram function
# ---------------------------------------------------------

def make_spec(x, fs):
    f, t, Z = signal.stft(
        x,
        fs=fs,
        nperseg=2048,
        noverlap=1536,
        window="hann"
    )

    mag = np.abs(Z)
    mag_db = 20*np.log10(mag + 1e-8)

    return f, t, mag_db

fa, ta, Sa = make_spec(actual, fs_actual)
fr, tr, Sr = make_spec(recon, fs_recon)

# ---------------------------------------------------------
# Save individual spectrograms
# ---------------------------------------------------------

plt.figure(figsize=(12,5))
plt.pcolormesh(
    ta, fa, Sa,
    shading="auto"
)
plt.ylim(0, 6000)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("ACTUAL AUDIO — Bass Music")
plt.colorbar(label="dB")
plt.tight_layout()

actual_spec = os.path.join(
    OUTDIR,
    "actual_bass_spectrogram.png"
)

plt.savefig(actual_spec, dpi=150)
plt.close()

plt.figure(figsize=(12,5))
plt.pcolormesh(
    tr, fr, Sr,
    shading="auto"
)
plt.ylim(0, 6000)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("REFLECTED-LIGHT RECONSTRUCTION")
plt.colorbar(label="dB")
plt.tight_layout()

recon_spec = os.path.join(
    OUTDIR,
    "reflection_reconstruction_spectrogram.png"
)

plt.savefig(recon_spec, dpi=150)
plt.close()

# ---------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------

fig, ax = plt.subplots(2, 1, figsize=(12,9))

ax[0].pcolormesh(
    ta, fa, Sa,
    shading="auto"
)
ax[0].set_ylim(0, 6000)
ax[0].set_title("ACTUAL BASS AUDIO")
ax[0].set_ylabel("Frequency (Hz)")

ax[1].pcolormesh(
    tr, fr, Sr,
    shading="auto"
)
ax[1].set_ylim(0, 6000)
ax[1].set_title("REFLECTED-LIGHT RECONSTRUCTION")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Frequency (Hz)")

plt.tight_layout()

comparison = os.path.join(
    OUTDIR,
    "actual_vs_reflection_spectrogram.png"
)

plt.savefig(comparison, dpi=150)
plt.close()

# ---------------------------------------------------------
# Compare low-frequency energy envelopes
# ---------------------------------------------------------

def low_band_envelope(x, fs, low=20, high=1000):

    sos = signal.butter(
        5,
        [low/(fs/2), high/(fs/2)],
        btype="bandpass",
        output="sos"
    )

    y = signal.sosfiltfilt(sos, x)

    analytic = signal.hilbert(y)
    env = np.abs(analytic)

    # smooth ~50 ms
    n = max(1, int(fs * 0.05))
    kernel = np.ones(n) / n
    env = np.convolve(env, kernel, mode="same")

    return env

ea = low_band_envelope(actual, fs_actual)
er = low_band_envelope(recon, fs_recon)

# Resample envelopes to common length
duration = min(
    len(ea)/fs_actual,
    len(er)/fs_recon
)

n = int(duration * 100)

ta2 = np.linspace(0, duration, n)
ea2 = np.interp(
    ta2,
    np.arange(len(ea))/fs_actual,
    ea
)

er2 = np.interp(
    ta2,
    np.arange(len(er))/fs_recon,
    er
)

ea2 = (ea2 - np.mean(ea2)) / (np.std(ea2)+1e-12)
er2 = (er2 - np.mean(er2)) / (np.std(er2)+1e-12)

corr = np.corrcoef(ea2, er2)[0,1]

print()
print("======================================")
print("RESULT")
print("======================================")
print("Actual duration:", len(actual)/fs_actual)
print("Reconstructed duration:", len(recon)/fs_recon)
print("Low-frequency envelope correlation:", corr)
print()
print("Saved:")
print(actual_spec)
print(recon_spec)
print(comparison)
print("======================================")

