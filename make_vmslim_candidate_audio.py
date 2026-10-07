import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt
from scipy.io.wavfile import write

INPUT = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_vmslim_phase.npz"

OUT_FULL = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_vmslim_full.wav"
OUT_CANDIDATE = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_vmslim_350_550Hz.wav"

FS = 48000

d = np.load(INPUT)

t = d["time"].astype(float)
x = d["signal"].astype(float)

valid = np.isfinite(t) & np.isfinite(x)
t = t[valid]
x = x[valid]

order = np.argsort(t)
t = t[order]
x = x[order]

# Remove duplicate timestamps
t, idx = np.unique(t, return_index=True)
x = x[idx]

# Remove DC
x -= np.mean(x)

# Uniform audio grid
tu = np.arange(t[0], t[-1], 1.0 / FS)

interp = interp1d(
    t,
    x,
    kind="linear",
    bounds_error=False,
    fill_value=0
)

audio = interp(tu)

# ------------------------------------------------
# Full reconstructed signal
# ------------------------------------------------

audio_full = audio / (np.max(np.abs(audio)) + 1e-12)
audio_full *= 0.95

write(
    OUT_FULL,
    FS,
    np.int16(audio_full * 32767)
)

# ------------------------------------------------
# Candidate band around expected acoustic tone
# ------------------------------------------------

sos = butter(
    6,
    [350, 550],
    btype="bandpass",
    fs=FS,
    output="sos"
)

candidate = sosfiltfilt(sos, audio)

candidate -= np.mean(candidate)

peak = np.max(np.abs(candidate))

if peak > 0:
    candidate /= peak

candidate *= 0.95

write(
    OUT_CANDIDATE,
    FS,
    np.int16(candidate * 32767)
)

print("==========================================")
print("VMSLIM CANDIDATE AUDIO")
print("==========================================")
print("Full reconstruction:")
print(OUT_FULL)

print()
print("350-550 Hz candidate:")
print(OUT_CANDIDATE)

print()
print("Listen specifically to:")
print("A15_vmslim_350_550Hz.wav")
print("==========================================")