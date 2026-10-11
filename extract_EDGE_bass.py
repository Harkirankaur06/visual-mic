import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.io import wavfile

OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")

data = np.load(
    OUT / "reflection_EDGE_tracking.npz"
)

edge = data["edge_profiles"].astype(np.float64)
deformation = data["deformation"].astype(np.float64)
fps = float(data["fps"])

ROW_DELAY = 7.7435e-6
FS = 48000

print("Edge matrix:", edge.shape)
print("FPS:", fps)

nframes, ncols = edge.shape

# ============================================================
# 1. FRAME-RATE EDGE DISPLACEMENT
# ============================================================

# Average the edge displacement spatially.
#
# We remove each column's static position first, so we're
# measuring actual movement of the reflected-light boundary.

frame_signal = np.median(
    deformation,
    axis=1
)

# Remove slow drift
frame_signal = detrend(frame_signal)

# ============================================================
# 2. ROLLING-SHUTTER EDGE SIGNAL
# ============================================================

times = []
values = []

for frame_idx in range(nframes):

    frame_time = frame_idx / fps

    rows = edge[frame_idx]
    vals = deformation[frame_idx]

    # Every x-column gives us:
    #
    #   edge displacement
    #   edge row
    #
    # The row gives the rolling-shutter acquisition time.

    for x in range(ncols):

        row = rows[x]
        value = vals[x]

        if not np.isfinite(row) or not np.isfinite(value):
            continue

        t = frame_time + row * ROW_DELAY

        times.append(t)
        values.append(value)

times = np.asarray(times)
values = np.asarray(values)

# Sort chronologically
order = np.argsort(times)

times = times[order]
values = values[order]

# Remove duplicate timestamps
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

print("Rolling-shutter samples:", len(values))

# Uniform high-rate reconstruction
t_uniform = np.arange(
    times[0],
    times[-1],
    1 / FS
)

rs_signal = np.interp(
    t_uniform,
    times,
    values
)

rs_signal = detrend(rs_signal)

# ============================================================
# BASS BAND FILTER
# ============================================================

def bass_filter(signal):

    sos = butter(
        5,
        [20, 250],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )

# Frame-rate signal must be upsampled first.
frame_t = np.arange(
    nframes
) / fps

frame_uniform = np.arange(
    frame_t[0],
    frame_t[-1],
    1 / FS
)

frame_signal_48k = np.interp(
    frame_uniform,
    frame_t,
    frame_signal
)

frame_signal_48k = bass_filter(
    frame_signal_48k
)

rs_signal = bass_filter(
    rs_signal
)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    x = x - np.mean(x)

    peak = np.max(
        np.abs(x)
    )

    if peak > 0:
        x = x / peak

    return x * 0.9


frame_signal_48k = normalize(
    frame_signal_48k
)

rs_signal = normalize(
    rs_signal
)

# ============================================================
# WRITE WAVs
# ============================================================

frame_wav = (
    OUT /
    "reflection_EDGE_frame_bass.wav"
)

rs_wav = (
    OUT /
    "reflection_EDGE_RS_bass.wav"
)

wavfile.write(
    frame_wav,
    FS,
    np.int16(
        np.clip(
            frame_signal_48k,
            -1,
            1
        ) * 32767
    )
)

wavfile.write(
    rs_wav,
    FS,
    np.int16(
        np.clip(
            rs_signal,
            -1,
            1
        ) * 32767
    )
)

# ============================================================
# SAVE NUMERICAL SIGNALS
# ============================================================

np.savez(
    OUT / "reflection_EDGE_audio_signals.npz",
    frame_signal=frame_signal_48k,
    rs_signal=rs_signal,
    fs=FS,
    fps=fps,
    row_delay=ROW_DELAY
)

print()
print("==============================================")
print("DONE")
print("==============================================")
print()
print("Frame-rate:")
print(frame_wav)
print()
print("Rolling-shutter:")
print(rs_wav)
print()
print("Band: 20-250 Hz")
print("==============================================")
