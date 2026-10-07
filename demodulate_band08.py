import numpy as np
from scipy.io import wavfile
from scipy import signal
from pathlib import Path

INPUT = Path(
    r"C:\Users\Harkiran\OneDrive\visual mic\speech_hybrid\candidate_08_band_06.wav"
)

OUTPUT = INPUT.parent / "band08_demodulated_speech.wav"

fs, x = wavfile.read(INPUT)

x = x.astype(np.float64)

if x.ndim > 1:
    x = np.mean(x, axis=1)

x -= np.mean(x)

# Normalize
x /= np.max(np.abs(x)) + 1e-12

# ---------------------------------------------------------
# 1. Remove very slow drift
# ---------------------------------------------------------

sos = signal.butter(
    4,
    30,
    btype="highpass",
    fs=fs,
    output="sos"
)

x = signal.sosfiltfilt(sos, x)

# ---------------------------------------------------------
# 2. Hilbert envelope
# ---------------------------------------------------------

analytic = signal.hilbert(x)

envelope = np.abs(analytic)

# Remove DC from envelope
envelope -= np.mean(envelope)

# ---------------------------------------------------------
# 3. Speech-rate modulation
# ---------------------------------------------------------

sos = signal.butter(
    4,
    [1.5, 20],
    btype="bandpass",
    fs=fs,
    output="sos"
)

speech_env = signal.sosfiltfilt(
    sos,
    envelope
)

# ---------------------------------------------------------
# 4. Smooth slightly
# ---------------------------------------------------------

sos = signal.butter(
    3,
    12,
    btype="lowpass",
    fs=fs,
    output="sos"
)

speech_env = signal.sosfiltfilt(
    sos,
    speech_env
)

# ---------------------------------------------------------
# 5. Normalize
# ---------------------------------------------------------

speech_env -= np.mean(speech_env)

speech_env /= (
    np.max(np.abs(speech_env)) +
    1e-12
)

speech_env *= 0.9

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

pcm = np.int16(
    np.clip(speech_env, -1, 1) * 32767
)

wavfile.write(
    str(OUTPUT),
    fs,
    pcm
)

print()
print("DONE")
print("Input :", INPUT)
print("Output:", OUTPUT)
print()