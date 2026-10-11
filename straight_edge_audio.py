import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.io import wavfile

OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")

D = np.load(
    OUT / "straight_edge_tracking.npz"
)

profiles = D["profiles"].astype(np.float64)
deformation = D["deformation"].astype(np.float64)
fps = float(D["fps"])

ROW_DELAY = 7.7435e-6
FS = 48000

nframes, ncols = deformation.shape

print("Edge field:", deformation.shape)
print("FPS:", fps)

# ============================================================
# BUILD ROLLING-SHUTTER TIME/VALUE SAMPLES
# ============================================================

times = []
values = []

for f in range(nframes):

    frame_time = f / fps

    rows = profiles[f]
    vals = deformation[f]

    valid = (
        np.isfinite(rows) &
        np.isfinite(vals)
    )

    rows = rows[valid]
    vals = vals[valid]

    t = frame_time + rows * ROW_DELAY

    times.extend(t)
    values.extend(vals)

times = np.asarray(times)
values = np.asarray(values)

order = np.argsort(times)

times = times[order]
values = values[order]

print("Raw RS samples:", len(values))

# ============================================================
# BIN INTO UNIFORM 48 kHz TIME GRID
# ============================================================

BIN = 1.0 / FS

bins = np.floor(
    (times - times[0]) / BIN
).astype(np.int64)

nbins = bins.max() + 1

signal = np.full(
    nbins,
    np.nan
)

# Efficient bin aggregation
for b in range(nbins):

    v = values[bins == b]

    if len(v):
        signal[b] = np.median(v)

good = np.isfinite(signal)

if good.sum() < 10:
    raise RuntimeError(
        "Too few valid rolling-shutter samples."
    )

x = np.arange(len(signal))

signal = np.interp(
    x,
    x[good],
    signal[good]
)

# ============================================================
# REMOVE SLOW DRIFT
# ============================================================

signal = detrend(signal)

# ============================================================
# BASS FILTER
# ============================================================

sos = butter(
    5,
    [20, 250],
    btype="bandpass",
    fs=FS,
    output="sos"
)

signal = sosfiltfilt(
    sos,
    signal
)

# ============================================================
# NORMALIZE
# ============================================================

signal -= np.mean(signal)

peak = np.max(
    np.abs(signal)
)

if peak > 0:
    signal /= peak

# ============================================================
# INCREASE LISTENING VOLUME
#
# 2.5x digital gain, followed by safe peak limiting.
# ============================================================

GAIN = 2.5

signal *= GAIN

# Safe normalization so it cannot clip
peak = np.max(
    np.abs(signal)
)

if peak > 0.98:
    signal *= 0.98 / peak

audio = np.int16(
    np.clip(
        signal,
        -1,
        1
    ) * 32767
)

# ============================================================
# WRITE WAV
# ============================================================

outfile = (
    OUT /
    "straight_edge_bass_RS_LOUD.wav"
)

wavfile.write(
    outfile,
    FS,
    audio
)

print()
print("======================================")
print("STRAIGHT EDGE AUDIO COMPLETE")
print("======================================")
print("Output:")
print(outfile)
print("Duration:", len(audio) / FS)
print("Samples:", len(audio))
print("Band: 20-250 Hz")
print("Listening gain:", GAIN, "x")
print("======================================")
