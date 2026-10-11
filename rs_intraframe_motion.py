# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\Harkiran\Downloads\newlightvibeexp\rs_edge_profile"

DATA = os.path.join(
    BASE,
    "row_profiles.npz"
)

OUT = os.path.join(
    BASE,
    "intraframe_motion"
)

os.makedirs(OUT, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

d = np.load(DATA)

profiles = d["profiles"].astype(float)
edge_positions = d["edge_positions"].astype(float)

fps = float(d["fps"])
row_delay = float(d["row_delay"])

nframes, nrows = profiles.shape

print("Profiles:", profiles.shape)
print("FPS:", fps)
print("Row delay:", row_delay)

# ============================================================
# BASIC ROW COORDINATES
# ============================================================

y0 = int(d["y0"])
y1 = int(d["y1"])

rows = np.arange(
    y0,
    y1
)

# ============================================================
# SMOOTH PROFILES
# ============================================================

P = np.zeros_like(
    profiles
)

for i in range(nframes):

    P[i] = signal.savgol_filter(
        profiles[i],
         nine := 9,
        2
    )

# ============================================================
# REFERENCE PROFILE
# ============================================================

reference = np.median(
    P,
    axis=0
)

reference -= np.mean(
    reference
)

# ============================================================
# ALIGN EACH PROFILE TO REFERENCE
#
# This removes ordinary frame-to-frame vertical movement.
# What remains is the shape change DURING readout.
# ============================================================

aligned = np.zeros_like(P)

shifts = np.zeros(
    nframes
)

for i in range(nframes):

    p = P[i].copy()

    p -= np.mean(p)

    # correlation
    corr = signal.correlate(
        p,
        reference,
        mode="full"
    )

    lags = signal.correlation_lags(
        len(p),
        len(reference),
        mode="full"
    )

    k = np.argmax(corr)

    lag = lags[k]

    # Limit ridiculous shifts
    lag = np.clip(
        lag,
        -30,
        30
    )

    shifts[i] = lag

    # Shift profile back onto reference
    x_old = np.arange(nrows)

    x_new = (
        x_old
        +
        lag
    )

    aligned[i] = np.interp(
        x_old,
        x_new,
        p,
        left=p[0],
        right=p[-1]
    )

# ============================================================
# RESIDUAL PROFILE
# ============================================================

residual = (
    aligned
    -
    reference[None, :]
)

# Remove frame-wide brightness offset
residual -= np.median(
    residual,
    axis=1,
    keepdims=True
)

# ============================================================
# IMPORTANT:
# USE ONLY THE EDGE REGIONS
#
# The center of the bright band is mostly static brightness.
# The leading/trailing edges contain displacement information.
# ============================================================

grad = np.gradient(
    reference
)

absgrad = np.abs(
    grad
)

# strongest edge locations
edge_order = np.argsort(
    absgrad
)[::-1]

edge_centers = []

for k in edge_order:

    y = rows[k]

    if all(
        abs(y - q) > 8
        for q in edge_centers
    ):

        edge_centers.append(y)

    if len(edge_centers) >= 2:
        break

edge_centers = sorted(
    edge_centers
)

print()
print("Detected profile edges:")

for y in edge_centers:
    print(y)

# ============================================================
# BUILD EDGE-WEIGHTED RESIDUAL
# ============================================================

weights = np.zeros(
    nrows
)

for ec in edge_centers:

    weights += np.exp(
        -0.5 *
        (
            (rows - ec) / 8.0
        ) ** 2
    )

weights /= (
    np.sum(weights)
    + 1e-12
)

# ============================================================
# EDGE RESIDUAL PER FRAME
# ============================================================

edge_residual = np.sum(
    residual *
    weights[None, :],
    axis=1
)

# ============================================================
# DIAGNOSTIC PLOT
# ============================================================

t = np.arange(
    nframes
) / fps

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    t,
    edge_residual,
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("Residual edge deformation")

plt.title(
    "Intra-frame Edge Residual"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "01_edge_residual.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# INTRA-FRAME TIME MAPPING
# ============================================================
#
# For each row:
#
# t = frame_time + row * row_delay
#
# But we DON'T concatenate raw profiles anymore.
#
# We use residual profile values.
# ============================================================

frame_time = (
    np.arange(nframes)
    / fps
)

row_times = (
    frame_time[:, None]
    +
    rows[None, :] * row_delay
)

# ============================================================
# REMOVE STATIC STRUCTURE
# ============================================================

R = residual.copy()

# Remove row-wise static residual
R -= np.median(
    R,
    axis=0,
    keepdims=True
)

# ============================================================
# EDGE WEIGHTING
# ============================================================

R *= weights[None, :]

# ============================================================
# FLATTEN
# ============================================================

times = row_times.ravel()

values = R.ravel()

order = np.argsort(
    times
)

times = times[order]
values = values[order]

# ============================================================
# IMPORTANT:
# REMOVE FRAME-LEVEL DISCONTINUITY
#
# Each frame contributes one readout interval.
# We high-pass the irregular sequence below 5 Hz only.
# This does NOT remove bass.
# ============================================================

# Remove very slow drift
values = signal.detrend(
    values,
    type="linear"
)

# ============================================================
# RESAMPLE
# ============================================================

SR = 48000

audio_t = np.arange(
    times[0],
    times[-1],
    1.0 / SR
)

rs = np.interp(
    audio_t,
    times,
    values
)

# ============================================================
# REMOVE 30-HZ FRAME COMB
#
# The previous experiment showed the frame-rate comb clearly.
# We explicitly notch 30,60,90,... only.
# ============================================================

clean = rs.copy()

for harmonic in range(
    1,
    31
):

    f0 = 30.00075632158794 * harmonic

    if f0 >= 800:
        break

    b, a = signal.iirnotch(
        f0,
        Q=20,
        fs=SR
    )

    clean = signal.filtfilt(
        b,
        a,
        clean
    )

# ============================================================
# AUDIO BAND
# ============================================================

b, a = signal.butter(
    4,
    [
        20 / (SR / 2),
        800 / (SR / 2)
    ],
    btype="band"
)

clean = signal.filtfilt(
    b,
    a,
    clean
)

# ============================================================
# NORMALIZE
# ============================================================

clean -= np.mean(
    clean
)

clean /= (
    np.max(
        np.abs(clean)
    )
    + 1e-12
)

clean *= 0.85

wav_path = os.path.join(
    OUT,
    "intraframe_RS_audio.wav"
)

wavfile.write(
    wav_path,
    SR,
    np.int16(
        clean * 32767
    )
)

# ============================================================
# SPECTRUM
# ============================================================

window = np.hanning(
    len(clean)
)

F = np.fft.rfft(
    clean * window
)

freq = np.fft.rfftfreq(
    len(clean),
    1 / SR
)

mag = np.abs(F)

mag /= (
    np.max(mag)
    + 1e-12
)

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    freq,
    mag
)

plt.xlim(
    0,
    1000
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized magnitude")

plt.title(
    "Intra-frame Rolling-Shutter Spectrum"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "02_intraframe_spectrum.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# SPECTROGRAM
# ============================================================

f, tt, S = signal.spectrogram(
    clean,
    fs=SR,
    nperseg=4096,
    noverlap=3072
)

plt.figure(
    figsize=(14, 6)
)

plt.pcolormesh(
    tt,
    f,
    10 * np.log10(
        S + 1e-12
    ),
    shading="auto"
)

plt.ylim(
    0,
    800
)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.title(
    "Intra-frame Rolling-Shutter Spectrogram"
)

plt.colorbar(
    label="dB"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT,
        "03_intraframe_spectrogram.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# PRINT PEAKS
# ============================================================

valid = (
    (freq >= 20)
    &
    (freq <= 800)
)

fv = freq[valid]
mv = mag[valid]

peaks, _ = signal.find_peaks(
    mv,
    distance=100
)

order = np.argsort(
    mv[peaks]
)[::-1]

print()
print("INTRA-FRAME COMPONENTS")

for k in order[:20]:

    p = peaks[k]

    print(
        f"{fv[p]:9.3f} Hz   "
        f"{mv[p]:.5f}"
    )

print()
print("==========================================")
print("INTRA-FRAME TEST COMPLETE")
print("==========================================")
print()
print("WAV:")
print(wav_path)
print()
print("Plots:")
print(
    os.path.join(
        OUT,
        "01_edge_residual.png"
    )
)
print(
    os.path.join(
        OUT,
        "02_intraframe_spectrum.png"
    )
)
print(
    os.path.join(
        OUT,
        "03_intraframe_spectrogram.png"
    )
)