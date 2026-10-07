import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, welch
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

INPUT = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_vmslim_phase.npz"
OUTPUT = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_vmslim_recovered.wav"
PLOT = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_vmslim_recovered.png"

AUDIO_FS = 48000

# --------------------------------------------------
# Load reconstruction
# --------------------------------------------------

d = np.load(INPUT)

t = d["time"].astype(np.float64)
x = d["signal"].astype(np.float64)

print("==========================================")
print("VMSLIM RECOVERED AUDIO")
print("==========================================")
print(f"Input samples : {len(x)}")
print(f"Time range    : {t[0]:.6f} - {t[-1]:.6f} s")
print(f"Duration      : {t[-1] - t[0]:.6f} s")
print(f"Row delay     : {float(d['row_delay'])*1e6:.4f} us")
print(f"Camera FPS    : {float(d['fps']):.6f}")

# --------------------------------------------------
# Remove invalid samples / sort timestamps
# --------------------------------------------------

valid = np.isfinite(t) & np.isfinite(x)

t = t[valid]
x = x[valid]

order = np.argsort(t)
t = t[order]
x = x[order]

# Remove duplicate timestamps
t_unique, idx = np.unique(t, return_index=True)
x_unique = x[idx]

t = t_unique
x = x_unique

# --------------------------------------------------
# Remove DC / slow drift
# --------------------------------------------------

x = x - np.mean(x)

# --------------------------------------------------
# Uniform audio time grid
# --------------------------------------------------

t_uniform = np.arange(t[0], t[-1], 1.0 / AUDIO_FS)

interp = interp1d(
    t,
    x,
    kind="linear",
    bounds_error=False,
    fill_value=0.0
)

audio = interp(t_uniform)

# --------------------------------------------------
# Band-limit reconstructed signal
#
# Keep a broad audio band first.
# We are NOT forcing the signal to 440 Hz.
# --------------------------------------------------

low = 50.0
high = 8000.0

sos = butter(
    6,
    [low, high],
    btype="bandpass",
    fs=AUDIO_FS,
    output="sos"
)

audio = sosfiltfilt(sos, audio)

# --------------------------------------------------
# Normalize safely
# --------------------------------------------------

audio = audio - np.mean(audio)

peak = np.max(np.abs(audio))

if peak > 0:
    audio = audio / peak

audio = 0.95 * audio

# --------------------------------------------------
# Save WAV
# --------------------------------------------------

audio_int16 = np.int16(audio * 32767)

write(
    OUTPUT,
    AUDIO_FS,
    audio_int16
)

print()
print("Saved WAV:")
print(OUTPUT)

# --------------------------------------------------
# Spectrogram / PSD
# --------------------------------------------------

freq, psd = welch(
    audio,
    fs=AUDIO_FS,
    nperseg=min(65536, len(audio))
)

# Only display useful audio range
mask = (freq >= 20) & (freq <= 10000)

plt.figure(figsize=(12, 5))
plt.semilogy(freq[mask], psd[mask] + 1e-15)
plt.axvline(440, linestyle="--", label="Expected 440 Hz")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("VMSlim Recovered Audio Spectrum")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(PLOT, dpi=150)
plt.close()

print("Saved spectrum:")
print(PLOT)

# --------------------------------------------------
# Basic diagnostics
# --------------------------------------------------

peak_idx = np.argmax(psd[mask])
peak_freq = freq[mask][peak_idx]

print()
print("==========================================")
print("AUDIO DIAGNOSTICS")
print("==========================================")
print(f"Dominant audio frequency: {peak_freq:.2f} Hz")
print(f"Peak amplitude          : {peak:.6e}")
print(f"Audio sample rate       : {AUDIO_FS} Hz")
print("==========================================")