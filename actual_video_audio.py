import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


# ============================================================
# INPUT / OUTPUT
# ============================================================

VIDEO = Path(
    r"C:\Users\Harkiran\Downloads\Mobile Devices\itsybitsy.mp4"
)

OUT_DIR = Path(
    r"C:\Users\Harkiran\Downloads\Mobile Devices"
) / "actual_video_audio"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIO_WAV = (
    OUT_DIR /
    "itsybitsy_original_audio.wav"
)

SPECTROGRAM = (
    OUT_DIR /
    "itsybitsy_original_audio_spectrogram.png"
)

REFERENCE_DATA = (
    OUT_DIR /
    "actual_audio_reference.npz"
)


# ============================================================
# FFMPEG
# ============================================================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("ORIGINAL VIDEO AUDIO EXTRACTION")
print("=" * 70)

print(
    f"Video:\n{VIDEO}"
)

print(
    f"\nFFmpeg:\n{FFMPEG}"
)

print()


# ============================================================
# CHECK VIDEO
# ============================================================

if not VIDEO.exists():

    raise FileNotFoundError(
        f"Video not found:\n{VIDEO}"
    )


# ============================================================
# EXTRACT ORIGINAL AUDIO
# ============================================================

command = [
    FFMPEG,
    "-y",
    "-i",
    str(VIDEO),
    "-vn",
    "-ac",
    "1",
    "-ar",
    "48000",
    "-c:a",
    "pcm_s16le",
    str(AUDIO_WAV)
]

print("Extracting original microphone audio...")

result = subprocess.run(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

if result.returncode != 0:

    print()
    print("FFmpeg failed.")
    print()
    print(result.stderr)

    raise RuntimeError(
        "Could not extract audio from video."
    )


print(
    f"Saved original audio:\n{AUDIO_WAV}"
)


# ============================================================
# LOAD AUDIO
# ============================================================

fs, audio = wavfile.read(
    str(AUDIO_WAV)
)

audio = audio.astype(
    np.float64
)

if audio.ndim > 1:

    audio = np.mean(
        audio,
        axis=1
    )

# Remove DC
audio -= np.mean(
    audio
)

# Normalize for analysis
peak = np.max(
    np.abs(audio)
)

if peak > 0:

    audio_normalized = (
        audio / peak
    )

else:

    audio_normalized = audio.copy()


duration = (
    len(audio_normalized) /
    fs
)


print()
print("=" * 70)
print("AUDIO INFORMATION")
print("=" * 70)

print(
    f"Sample rate : {fs} Hz"
)

print(
    f"Samples     : {len(audio_normalized)}"
)

print(
    f"Duration    : {duration:.3f} s"
)

print(
    f"Peak        : {np.max(np.abs(audio)):.2f}"
)


# ============================================================
# RMS / AMPLITUDE ENVELOPE
# ============================================================

# 30 ms RMS window
WINDOW_MS = 30

window = max(
    1,
    int(
        fs *
        WINDOW_MS /
        1000
    )
)

rms = np.sqrt(
    signal.fftconvolve(
        audio_normalized ** 2,
        np.ones(window) / window,
        mode="same"
    )
)

rms_db = (
    20 *
    np.log10(
        rms + 1e-10
    )
)

rms_time = (
    np.arange(
        len(rms)
    ) / fs
)


# ============================================================
# SPECTROGRAM
# ============================================================

print()
print("Creating spectrogram...")

f, t, Sxx = signal.spectrogram(
    audio_normalized,
    fs=fs,
    window="hann",
    nperseg=4096,
    noverlap=3072,
    scaling="density",
    mode="magnitude"
)

Sxx_db = (
    20 *
    np.log10(
        Sxx + 1e-10
    )
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 9),
    gridspec_kw={
        "height_ratios": [3, 1]
    }
)


# ============================================================
# SPECTROGRAM
# ============================================================

im = axes[0].pcolormesh(
    t,
    f,
    Sxx_db,
    shading="auto"
)

axes[0].set_ylim(
    0,
    5000
)

axes[0].set_ylabel(
    "Frequency (Hz)"
)

axes[0].set_title(
    "Actual Microphone Audio from itsybitsy.mp4"
)

fig.colorbar(
    im,
    ax=axes[0],
    label="Magnitude (dB)"
)


# ============================================================
# RMS ENVELOPE
# ============================================================

axes[1].plot(
    rms_time,
    rms_db
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "RMS (dB)"
)

axes[1].set_title(
    "Actual Audio Amplitude Envelope"
)

axes[1].grid(
    True,
    alpha=0.25
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()

plt.savefig(
    SPECTROGRAM,
    dpi=200
)

plt.close()


print(
    f"Saved spectrogram:\n{SPECTROGRAM}"
)


# ============================================================
# SAVE REFERENCE DATA
# ============================================================

np.savez(
    REFERENCE_DATA,

    # Original audio
    audio=audio_normalized,

    # Audio information
    fs=fs,

    # RMS envelope
    rms=rms,

    rms_db=rms_db,

    rms_time=rms_time,

    # Spectrogram
    spectrogram=Sxx_db,

    frequencies=f,

    times=t
)


print(
    f"Saved reference data:\n{REFERENCE_DATA}"
)


# ============================================================
# ALSO CREATE A SIMPLE FULL-SPECTRUM FIGURE
# ============================================================

# This one is useful because the main spectrogram is limited
# to 0-5000 Hz.

fft = np.fft.rfft(
    audio_normalized
)

freq_fft = np.fft.rfftfreq(
    len(audio_normalized),
    1 / fs
)

power = (
    np.abs(fft) ** 2
)

power_db = (
    10 *
    np.log10(
        power + 1e-12
    )
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    freq_fft,
    power_db
)

plt.xlim(
    0,
    8000
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Power (dB)"
)

plt.title(
    "Actual Microphone Audio — Frequency Spectrum"
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

SPECTRUM = (
    OUT_DIR /
    "itsybitsy_original_audio_spectrum.png"
)

plt.savefig(
    SPECTRUM,
    dpi=180
)

plt.close()


print(
    f"Saved spectrum:\n{SPECTRUM}"
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print("Output folder:")
print(
    OUT_DIR
)

print()
print("Files:")
print(
    f"  {AUDIO_WAV.name}"
)

print(
    f"  {SPECTROGRAM.name}"
)

print(
    f"  {SPECTRUM.name}"
)

print(
    f"  {REFERENCE_DATA.name}"
)

print()
print("Open this first:")
print(
    SPECTROGRAM
)

print()