# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"

OUT_DIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp\rs_edge_profile"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CAMERA
# ============================================================

FPS = 30.00075632158794

ROW_DELAY = 7.7435e-6

SR = 48000

# ============================================================
# ROI
#
# Your reflected line was around y = 775-850.
# We deliberately use a larger vertical region so that
# the rolling-shutter edge is completely contained.
# ============================================================

X0 = 350
X1 = 1439

Y0 = 740
Y1 = 880

# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

nframes = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print("Frames:", nframes)
print("Resolution:", width, height)
print("FPS:", FPS)

# ============================================================
# STORE ROW PROFILES
# ============================================================

profiles = []

edge_positions = []

frame_times = []

# ============================================================
# PROCESS
# ============================================================

for fi in range(nframes):

    ok, frame = cap.read()

    if not ok:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[
        Y0:Y1,
        X0:X1
    ].astype(
        np.float32
    )

    # --------------------------------------------------------
    # Average across X.
    #
    # This suppresses individual water texture and keeps
    # the horizontal reflection structure.
    # --------------------------------------------------------

    profile = np.mean(
        roi,
        axis=1
    )

    # Smooth only slightly
    profile = cv2.GaussianBlur(
        profile.reshape(-1, 1),
        (1, 0),
        1.2
    ).ravel()

    profiles.append(
        profile
    )

    # --------------------------------------------------------
    # Vertical gradient
    # --------------------------------------------------------

    grad = np.gradient(
        profile
    )

    # strongest positive/negative edge
    # Use absolute gradient because polarity may change.
    absgrad = np.abs(grad)

    # restrict to central expected region
    yy = np.arange(
        len(profile)
    )

    mask = (
        (yy >= 10)
        &
        (yy < len(profile) - 10)
    )

    score = absgrad.copy()

    score[~mask] = 0

    k = np.argmax(
        score
    )

    # --------------------------------------------------------
    # SUBPIXEL EDGE POSITION
    # --------------------------------------------------------

    subpixel = float(k)

    if 1 <= k < len(grad) - 1:

        a = absgrad[k - 1]
        b = absgrad[k]
        c = absgrad[k + 1]

        denom = (
            a - 2 * b + c
        )

        if abs(denom) > 1e-12:

            delta = (
                0.5
                *
                (a - c)
                /
                denom
            )

            if abs(delta) <= 1:

                subpixel += delta

    edge_positions.append(
        Y0 + subpixel
    )

    frame_times.append(
        fi / FPS
    )

cap.release()

profiles = np.asarray(
    profiles
)

edge_positions = np.asarray(
    edge_positions
)

frame_times = np.asarray(
    frame_times
)

print()
print("Profile matrix:", profiles.shape)

print(
    "Edge position range:",
    edge_positions.min(),
    edge_positions.max()
)

# ============================================================
# SAVE PROFILE DATA
# ============================================================

np.savez(
    os.path.join(
        OUT_DIR,
        "row_profiles.npz"
    ),
    profiles=profiles,
    edge_positions=edge_positions,
    frame_times=frame_times,
    y0=Y0,
    y1=Y1,
    x0=X0,
    x1=X1,
    fps=FPS,
    row_delay=ROW_DELAY
)

# ============================================================
# PLOT EDGE POSITION
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    frame_times,
    edge_positions,
    linewidth=1
)

plt.xlabel("Frame time (s)")
plt.ylabel("Edge row")

plt.title(
    "Detected Reflection Edge Position"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "01_edge_position.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# PLOT ROW PROFILES
# ============================================================

plt.figure(figsize=(14, 7))

# Show selected frames
show_frames = np.linspace(
    0,
    len(profiles) - 1,
    min(12, len(profiles))
).astype(int)

rows = np.arange(
    Y0,
    Y1
)

for fi in show_frames:

    p = profiles[fi]

    p = (
        p - np.mean(p)
    )

    p /= (
        np.std(p) + 1e-12
    )

    plt.plot(
        rows,
        p,
        alpha=0.55
    )

plt.xlabel("Image row")
plt.ylabel("Normalized brightness")

plt.title(
    "Vertical Brightness Profiles Around Reflection"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "02_row_profiles.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# PROFILE DEFORMATION
# ============================================================

profile_mean = np.mean(
    profiles,
    axis=0
)

profile_deformation = (
    profiles
    -
    profile_mean[None, :]
)

# ============================================================
# ROW-WISE TEMPORAL SIGNAL
# ============================================================
#
# Every image row corresponds to a different capture time.
#
# For each frame we therefore retain the complete row profile.
#
# We remove the static mean profile, but DO NOT remove
# the temporal variation.
# ============================================================

dynamic = profile_deformation.copy()

# remove slow frame-wide brightness offset
dynamic -= np.mean(
    dynamic,
    axis=1,
    keepdims=True
)

# ============================================================
# BUILD ROLLING-SHUTTER TIME AXIS
# ============================================================

nrows = dynamic.shape[1]

row_indices = np.arange(
    Y0,
    Y1
)

# Frame start times
base_times = (
    np.arange(
        len(dynamic)
    )
    / FPS
)

# absolute row capture time
row_times = (
    base_times[:, None]
    +
    row_indices[None, :] * ROW_DELAY
)

# ============================================================
# FLATTEN INTO IRREGULAR HIGH-RATE SIGNAL
# ============================================================

times = row_times.ravel()

values = dynamic.ravel()

order = np.argsort(
    times
)

times = times[order]
values = values[order]

# ============================================================
# REMOVE DUPLICATE TIMES
# ============================================================

dt = np.diff(
    times
)

keep = np.concatenate(
    (
        [True],
        dt > 1e-10
    )
)

times = times[keep]
values = values[keep]

# ============================================================
# RESAMPLE
# ============================================================

audio_t = np.arange(
    times[0],
    times[-1],
    1.0 / SR
)

rs_signal = np.interp(
    audio_t,
    times,
    values
)

rs_signal -= np.mean(
    rs_signal
)

# ============================================================
# LOW-PASS / AUDIO BAND
# ============================================================

# We are NOT using this as proof of audio.
# This is just a diagnostic audible rendering.

b, a = signal.butter(
    4,
    [
        20 / (SR / 2),
        800 / (SR / 2)
    ],
    btype="band"
)

audio = signal.filtfilt(
    b,
    a,
    rs_signal
)

# ============================================================
# NORMALIZE
# ============================================================

audio /= (
    np.max(
        np.abs(audio)
    )
    + 1e-12
)

audio *= 0.8

wav_path = os.path.join(
    OUT_DIR,
    "rs_edge_profile_diagnostic.wav"
)

wavfile.write(
    wav_path,
    SR,
    np.int16(
        audio * 32767
    )
)

# ============================================================
# SPECTRUM
# ============================================================

window = np.hanning(
    len(rs_signal)
)

F = np.fft.rfft(
    rs_signal * window
)

freq = np.fft.rfftfreq(
    len(rs_signal),
    1 / SR
)

mag = np.abs(F)

# Normalize for plotting
mag /= (
    np.max(mag) + 1e-12
)

plt.figure(figsize=(14, 5))

plt.plot(
    freq,
    mag,
    linewidth=0.7
)

plt.xlim(
    0,
    1000
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized magnitude")

plt.title(
    "Rolling-Shutter Row-Profile Spectrum"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "03_rs_spectrum.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# SPECTROGRAM
# ============================================================

nperseg = 4096
noverlap = 3072

f, tt, S = signal.spectrogram(
    rs_signal,
    fs=SR,
    window="hann",
    nperseg=nperseg,
    noverlap=noverlap,
    scaling="density"
)

plt.figure(figsize=(14, 6))

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
    1000
)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.title(
    "Rolling-Shutter Row-Profile Spectrogram"
)

plt.colorbar(
    label="dB"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "04_rs_spectrogram.png"
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
    (freq <= 1000)
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
print("STRONGEST RS COMPONENTS")

for k in order[:20]:

    p = peaks[k]

    print(
        f"{fv[p]:9.3f} Hz   "
        f"{mv[p]:.5f}"
    )

# ============================================================
# DONE
# ============================================================

print()
print("============================================")
print("ROLLING-SHUTTER EDGE-PROFILE TEST COMPLETE")
print("============================================")
print()
print("Diagnostic WAV:")
print(wav_path)
print()
print("Open these:")
print(
    os.path.join(
        OUT_DIR,
        "02_row_profiles.png"
    )
)
print(
    os.path.join(
        OUT_DIR,
        "03_rs_spectrum.png"
    )
)
print(
    os.path.join(
        OUT_DIR,
        "04_rs_spectrogram.png"
    )
)
print("============================================")