import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, welch
from scipy.io.wavfile import write

INPUT = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_vmslim_phase.npz"

OUTPUT = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_500Hz_water_candidate.wav"

FS = 48000

# --------------------------------------------------
# Load raw VMSlim rolling-shutter reconstruction
# --------------------------------------------------

d = np.load(INPUT)

t = d["time"].astype(np.float64)
x = d["signal"].astype(np.float64)

valid = np.isfinite(t) & np.isfinite(x)
t = t[valid]
x = x[valid]

order = np.argsort(t)
t = t[order]
x = x[order]

t, idx = np.unique(t, return_index=True)
x = x[idx]

print("==========================================")
print("WATER VMSLIM AUDIO RECONSTRUCTION")
print("==========================================")
print(f"Samples    : {len(x)}")
print(f"Duration   : {t[-1]-t[0]:.6f} s")
print(f"Row delay  : {float(d['row_delay'])*1e6:.4f} us")
print(f"FPS        : {float(d['fps']):.6f}")

# --------------------------------------------------
# Remove mean only
# --------------------------------------------------

x -= np.mean(x)

# --------------------------------------------------
# Uniform resampling
# --------------------------------------------------

tu = np.arange(t[0], t[-1], 1.0 / FS)

f_interp = interp1d(
    t,
    x,
    kind="linear",
    bounds_error=False,
    fill_value=0.0
)

xu = f_interp(tu)

# --------------------------------------------------
# IMPORTANT:
# Focus on the region around the observed
# ~510 Hz component.
#
# This is NOT claiming that 510 Hz is correct.
# It is a diagnostic listening band.
# --------------------------------------------------

LOW = 350.0
HIGH = 700.0

sos = butter(
    6,
    [LOW, HIGH],
    btype="bandpass",
    fs=FS,
    output="sos"
)

audio = sosfiltfilt(sos, xu)

# --------------------------------------------------
# Remove residual DC
# --------------------------------------------------

audio -= np.mean(audio)

# --------------------------------------------------
# Normalize
# --------------------------------------------------

peak = np.max(np.abs(audio))

if peak > 0:
    audio /= peak

audio *= 0.95

# --------------------------------------------------
# Save WAV
# --------------------------------------------------

write(
    OUTPUT,
    FS,
    np.int16(audio * 32767)
)

print()
print("Saved:")
print(OUTPUT)

# --------------------------------------------------
# Diagnostic spectrum AFTER filtering
# --------------------------------------------------

f, p = welch(
    audio,
    fs=FS,
    nperseg=min(65536, len(audio))
)

mask = (f >= 300) & (f <= 800)

idx = np.argmax(p[mask])

peak_freq = f[mask][idx]

print()
print("==========================================")
print("FILTERED AUDIO DIAGNOSTIC")
print("==========================================")
print(f"Dominant frequency: {peak_freq:.2f} Hz")
print("Band: 350-700 Hz")
print("==========================================")