import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.io import wavfile

OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")

DATA = np.load(
    OUT / "reflection_continuous_tracking.npz"
)

profiles = DATA["profiles"].astype(np.float64)
deformation = DATA["deformation"].astype(np.float64)
fps = float(DATA["fps"])

ROW_DELAY = 7.7435e-6
FS = 48000

print("Tracking:", profiles.shape)
print("FPS:", fps)

# ============================================================
# IMPORTANT:
# Each tracked point has a slightly different rolling-shutter
# acquisition time depending on its vertical position.
#
# Instead of treating every x-column independently, group
# measurements by ROW and robustly average across x.
# This suppresses the tiny texture fluctuations that were
# producing the drill-like artifact.
# ============================================================

nframes, ncols = deformation.shape

times = []
values = []

for frame_idx in range(nframes):

    frame_time = frame_idx / fps

    rows = np.round(
        profiles[frame_idx]
    ).astype(int)

    vals = deformation[frame_idx]

    # Group all x-columns having approximately the same row.
    for row in np.unique(rows):

        mask = rows == row

        if np.sum(mask) < 3:
            continue

        # Robust spatial aggregation
        value = np.median(vals[mask])

        # Rolling-shutter acquisition time
        t = frame_time + row * ROW_DELAY

        times.append(t)
        values.append(value)

times = np.asarray(times)
values = np.asarray(values)

# Sort chronologically
order = np.argsort(times)

times = times[order]
values = values[order]

# Remove duplicate timestamps by averaging them
unique_t, inverse = np.unique(
    times,
    return_inverse=True
)

sum_v = np.zeros(len(unique_t))
count_v = np.zeros(len(unique_t))

np.add.at(sum_v, inverse, values)
np.add.at(count_v, inverse, 1)

values = sum_v / np.maximum(count_v, 1)
times = unique_t

print("RS samples:", len(values))
print("Time span:", times[0], "to", times[-1])

# ============================================================
# UNIFORM HIGH-RATE SIGNAL
# ============================================================

t_uniform = np.arange(
    times[0],
    times[-1],
    1 / FS
)

signal = np.interp(
    t_uniform,
    times,
    values
)

signal = detrend(signal)

# ============================================================
# BASS BAND
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

# Remove remaining DC
signal -= np.mean(signal)

# ============================================================
# NORMALIZE
# ============================================================

peak = np.max(np.abs(signal))

if peak > 0:
    signal /= peak

# Conservative level
signal *= 0.9

audio = np.int16(
    np.clip(signal, -1, 1) * 32767
)

# ============================================================
# WRITE WAV
# ============================================================

outfile = OUT / "reflection_bass_RS_recovered.wav"

wavfile.write(
    outfile,
    FS,
    audio
)

print("\n========================================")
print("DONE")
print("Output:")
print(outfile)
print("Duration:", len(audio) / FS)
print("Band: 20-250 Hz")
print("========================================")
