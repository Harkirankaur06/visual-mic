# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch
from scipy.io.wavfile import write


# ============================================================
# PATHS
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\newlightvibeexp\MANUAL_LINE_tracking.mp4"

OUT = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

PROFILE_PLOT = os.path.join(
    OUT, "FINAL_row_profiles.png"
)

EDGE_PLOT = os.path.join(
    OUT, "FINAL_edge_profiles.png"
)

WAVE_PLOT = os.path.join(
    OUT, "FINAL_RS_waveform.png"
)

SPEC_PLOT = os.path.join(
    OUT, "FINAL_RS_spectrum.png"
)

WAV_OUT = os.path.join(
    OUT, "FINAL_RS_reconstructed.wav"
)


# ============================================================
# CALIBRATED CAMERA PARAMETERS
# ============================================================

FPS = 30.00075632158794

ROW_DELAY = 7.7435e-6

AUDIO_FS = 48000


# ============================================================
# ANALYSIS REGION
# ============================================================

# Your manually selected line was approximately:
#
# X = 775 ... 979
# Y = 799 ... 805
#
# We use a much wider vertical region around it because
# the acoustic movement is in Y.

X0 = 775
X1 = 979

Y0_OFFSET = -60
Y1_OFFSET = +60


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open MANUAL_LINE_tracking.mp4"
    )


fps_video = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

nframes = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print()
print("================================================")
print("FINAL ROLLING-SHUTTER EXPERIMENT")
print("================================================")
print()

print("Video:", VIDEO)
print("FPS:", fps_video)
print("Resolution:", width, height)
print("Frames:", nframes)

print()
print("Analysis X:", X0, "to", X1)


# ============================================================
# STORAGE
# ============================================================

profiles = []
edge_profiles = []
green_y = []

frame_index = 0


# ============================================================
# PROCESS EACH FRAME
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------------------------
    # Detect green reference line.
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    green_mask = cv2.inRange(
        hsv,
        np.array([35, 60, 60]),
        np.array([90, 255, 255])
    )


    # Restrict green detection to our expected region.

    roi_green = green_mask[
        max(0, 730):
        min(height, 880),
        X0:X1
    ]

    gy, gx = np.where(
        roi_green > 0
    )


    if len(gy) > 20:

        reference_y = (
            np.median(gy) + 730
        )

    else:

        reference_y = 802


    green_y.append(
        reference_y
    )


    # --------------------------------------------------------
    # Vertical analysis region.
    # --------------------------------------------------------

    ya = max(
        0,
        int(reference_y + Y0_OFFSET)
    )

    yb = min(
        height,
        int(reference_y + Y1_OFFSET)
    )


    crop = frame[
        ya:yb,
        X0:X1
    ]


    # --------------------------------------------------------
    # Use RED + BLUE channels.
    #
    # Why?
    #
    # The tracking line is green.
    # Green overlay strongly affects G.
    # R and B preserve much more of the underlying image.
    # --------------------------------------------------------

    B = crop[:, :, 0].astype(np.float32)

    R = crop[:, :, 2].astype(np.float32)

    non_green_image = (
        R + B
    ) / 2.0


    # --------------------------------------------------------
    # Remove remaining green-overlay pixels.
    # --------------------------------------------------------

    green_crop = green_mask[
        ya:yb,
        X0:X1
    ]


    valid = (
        green_crop == 0
    )


    profile = np.zeros(
        yb - ya,
        dtype=np.float32
    )


    for r in range(
        yb - ya
    ):

        values = non_green_image[
            r,
            valid[r]
        ]

        if len(values) > 5:

            profile[r] = np.median(
                values
            )

        else:

            profile[r] = np.nan


    # Interpolate missing rows.

    good = np.isfinite(profile)

    if np.sum(good) > 5:

        profile = np.interp(
            np.arange(len(profile)),
            np.flatnonzero(good),
            profile[good]
        )

    else:

        profile[:] = 0


    # Smooth only slightly.

    profile = cv2.GaussianBlur(
        profile.reshape(-1, 1),
        (1, 5),
        0
    ).flatten()


    # --------------------------------------------------------
    # Vertical edge.
    # --------------------------------------------------------

    gradient = np.gradient(
        profile
    )

    edge_profiles.append(
        gradient
    )

    profiles.append(
        profile
    )

    frame_index += 1


cap.release()


# ============================================================
# ARRAYS
# ============================================================

profiles = np.asarray(
    profiles,
    dtype=np.float64
)

edge_profiles = np.asarray(
    edge_profiles,
    dtype=np.float64
)

green_y = np.asarray(
    green_y,
    dtype=np.float64
)


print()
print("Profile matrix:", profiles.shape)
print("Edge matrix:", edge_profiles.shape)

print(
    "Green reference Y:",
    np.min(green_y),
    "to",
    np.max(green_y)
)


# ============================================================
# REMOVE STATIC PROFILE
# ============================================================

# This is important.
#
# We do NOT use the absolute brightness profile.
#
# We calculate how the row profile changes relative to
# its normal/static shape.

reference_profile = np.median(
    profiles,
    axis=0
)

residual = (
    profiles
    - reference_profile[None, :]
)


# Same idea for edge profile.

reference_edge = np.median(
    edge_profiles,
    axis=0
)

edge_residual = (
    edge_profiles
    - reference_edge[None, :]
)


# ============================================================
# PLOT ROW PROFILES
# ============================================================

plt.figure(figsize=(14, 6))

# Show selected frames.

for i in np.linspace(
    0,
    len(profiles) - 1,
    8
).astype(int):

    plt.plot(
        profiles[i],
        label=f"Frame {i}"
    )


plt.xlabel("Vertical row inside analysis region")
plt.ylabel("R+B intensity")
plt.title(
    "Vertical light profiles around manually marked reflection line"
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    PROFILE_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# PLOT EDGE PROFILES
# ============================================================

plt.figure(figsize=(14, 6))

for i in np.linspace(
    0,
    len(edge_profiles) - 1,
    8
).astype(int):

    plt.plot(
        edge_profiles[i],
        label=f"Frame {i}"
    )


plt.xlabel("Vertical row")
plt.ylabel("Vertical brightness gradient")

plt.title(
    "Vertical edge profiles around the vibrating reflection"
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    EDGE_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# ROLLING-SHUTTER RECONSTRUCTION
# ============================================================

#
# Every row inside each frame has a different capture time.
#
# t = frame/FPS + row*ROW_DELAY
#
# We now use the RESIDUAL EDGE signal at every row.
#

rs_times = []
rs_values = []


for f in range(
    edge_residual.shape[0]
):

    frame_time = (
        f / FPS
    )


    for r in range(
        edge_residual.shape[1]
    ):

        value = edge_residual[
            f,
            r
        ]

        if not np.isfinite(value):
            continue


        # Actual image row.
        #
        # Reference line is approximately at green_y[f].
        # r is relative to reference_y - 60.

        actual_row = (
            green_y[f]
            + Y0_OFFSET
            + r
        )


        capture_time = (
            frame_time
            + actual_row * ROW_DELAY
        )


        rs_times.append(
            capture_time
        )

        rs_values.append(
            value
        )


rs_times = np.asarray(
    rs_times,
    dtype=np.float64
)

rs_values = np.asarray(
    rs_values,
    dtype=np.float64
)


# ============================================================
# CLEAN
# ============================================================

good = (
    np.isfinite(rs_times)
    &
    np.isfinite(rs_values)
)

rs_times = rs_times[good]
rs_values = rs_values[good]


order = np.argsort(
    rs_times
)

rs_times = rs_times[order]
rs_values = rs_values[order]


print()
print("Rolling-shutter samples:", len(rs_values))

print(
    "Time range:",
    rs_times[0],
    "to",
    rs_times[-1]
)


# ============================================================
# REMOVE DUPLICATE TIMES
# ============================================================

unique_times, inverse = np.unique(
    rs_times,
    return_inverse=True
)

unique_values = np.zeros(
    len(unique_times)
)

counts = np.zeros(
    len(unique_times)
)

np.add.at(
    unique_values,
    inverse,
    rs_values
)

np.add.at(
    counts,
    inverse,
    1
)

unique_values /= np.maximum(
    counts,
    1
)


# ============================================================
# INTERPOLATE
# ============================================================

audio_time = np.arange(
    unique_times[0],
    unique_times[-1],
    1.0 / AUDIO_FS
)


audio_raw = np.interp(
    audio_time,
    unique_times,
    unique_values
)


audio_raw = np.nan_to_num(
    audio_raw
)


audio_raw -= np.mean(
    audio_raw
)


print(
    "High-rate samples:",
    len(audio_raw)
)

print(
    "Duration:",
    len(audio_raw) / AUDIO_FS
)

print(
    "Raw RMS:",
    np.sqrt(
        np.mean(
            audio_raw ** 2
        )
    )
)


# ============================================================
# REMOVE 30-HZ CAMERA COMB
# ============================================================

audio_clean = audio_raw.copy()


# Notches at frame-rate harmonics.

from scipy.signal import iirnotch


for harmonic in range(
    1,
    27
):

    f0 = harmonic * FPS

    if f0 >= AUDIO_FS / 2:
        break

    b_notch, a_notch = iirnotch(
        f0,
        30,
        AUDIO_FS
    )

    audio_clean = filtfilt(
        b_notch,
        a_notch,
        audio_clean
    )


# ============================================================
# AUDIO BAND
# ============================================================

nyquist = AUDIO_FS / 2


b, a = butter(
    4,
    [
        20 / nyquist,
        250 / nyquist
    ],
    btype="band"
)


audio_clean = filtfilt(
    b,
    a,
    audio_clean
)


audio_clean = np.nan_to_num(
    audio_clean
)


# ============================================================
# WAV
# ============================================================

peak = np.max(
    np.abs(audio_clean)
)

print()
print(
    "After filtering RMS:",
    np.sqrt(
        np.mean(
            audio_clean ** 2
        )
    )
)

print(
    "After filtering peak:",
    peak
)


if peak > 1e-12:

    audio_out = (
        audio_clean / peak
    )

else:

    audio_out = (
        audio_clean
    )


audio_out *= 0.9


write(
    WAV_OUT,
    AUDIO_FS,
    np.int16(
        np.clip(
            audio_out,
            -1,
            1
        ) * 32767
    )
)


# ============================================================
# WAVEFORM PLOT
# ============================================================

seconds = min(
    7,
    len(audio_out) / AUDIO_FS
)

N = int(
    seconds * AUDIO_FS
)


plt.figure(figsize=(14, 5))

plt.plot(
    np.arange(N) / AUDIO_FS,
    audio_out[:N],
    linewidth=0.5
)

plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")

plt.title(
    "Rolling-shutter reconstructed waveform from marked reflection"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    WAVE_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# SPECTRUM
# ============================================================

freq, power = welch(
    audio_out,
    fs=AUDIO_FS,
    nperseg=min(
        65536,
        len(audio_out)
    )
)


mask = (
    (freq >= 1)
    &
    (freq <= 1000)
    &
    np.isfinite(power)
)


plt.figure(figsize=(14, 5))

plt.plot(
    freq[mask],
    power[mask]
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.title(
    "Spectrum of rolling-shutter reconstructed signal"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    SPEC_PLOT,
    dpi=200
)

plt.close()


# ============================================================
# TOP FREQUENCIES
# ============================================================

indices = np.argsort(
    power[mask]
)[-20:][::-1]


print()
print("================================================")
print("TOP FREQUENCIES")
print("================================================")

for i in indices:

    print(
        f"{freq[mask][i]:8.2f} Hz   "
        f"{power[mask][i]:.8e}"
    )


print()
print("================================================")
print("FINAL OUTPUT")
print("================================================")

print()
print("Row profiles:")
print(PROFILE_PLOT)

print()
print("Edge profiles:")
print(EDGE_PLOT)

print()
print("Waveform:")
print(WAVE_PLOT)

print()
print("Spectrum:")
print(SPEC_PLOT)

print()
print("WAV:")
print(WAV_OUT)

print()
print("DONE")