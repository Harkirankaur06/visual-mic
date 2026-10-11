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

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"

OUT = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

TRACKING_VIDEO = os.path.join(
    OUT, "MANUAL_LINE_tracking.mp4"
)

NPZ_OUT = os.path.join(
    OUT, "MANUAL_LINE_tracking.npz"
)

WAVE_PLOT = os.path.join(
    OUT, "MANUAL_LINE_waveform.png"
)

FAST_PLOT = os.path.join(
    OUT, "MANUAL_LINE_fast_motion.png"
)

SPECTRUM_PLOT = os.path.join(
    OUT, "MANUAL_LINE_spectrum.png"
)

WAV_OUT = os.path.join(
    OUT, "MANUAL_LINE_reconstructed.wav"
)

# ============================================================
# CAMERA
# ============================================================

ROW_DELAY = 7.7435e-6
AUDIO_FS = 48000

# Search vertical movement only.
SEARCH_Y = 25

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open light.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("VIDEO")
print("FPS:", fps)
print("Resolution:", width, height)
print("Frames:", nframes)

# Read first frame.
ret, first = cap.read()

if not ret:
    raise RuntimeError("Could not read first frame")

# ============================================================
# MANUAL LINE DRAWING
# ============================================================

drawing = False
line_points = []

display = first.copy()

print()
print("==============================================")
print("DRAW THE ENTIRE REFLECTION LINE")
print("==============================================")
print()
print("Hold LEFT mouse button and draw along")
print("the white horizontal reflected-light line.")
print()
print("You can draw from left to right.")
print()
print("Press ENTER when finished.")
print("Press R to redraw.")
print("Press ESC to cancel.")
print()

def mouse_callback(event, x, y, flags, param):

    global drawing
    global line_points
    global display

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        line_points.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            line_points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False


window = "DRAW REFLECTION LINE"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1200, 800)

cv2.setMouseCallback(window, mouse_callback)

while True:

    display = first.copy()

    # Draw the line being selected.
    if len(line_points) > 1:

        pts = np.asarray(
            line_points,
            dtype=np.int32
        )

        cv2.polylines(
            display,
            [pts],
            False,
            (0, 255, 0),
            3
        )

    cv2.imshow(window, display)

    key = cv2.waitKey(20) & 0xFF

    if key == 13:       # ENTER
        break

    if key == ord("r"):

        line_points = []

    if key == 27:       # ESC

        cv2.destroyAllWindows()
        cap.release()
        raise SystemExit

cv2.destroyWindow(window)

if len(line_points) < 20:
    cap.release()
    raise RuntimeError(
        "Not enough line points were drawn."
    )

line_points = np.asarray(
    line_points,
    dtype=np.float32
)

print()
print("Manually drawn points:", len(line_points))

# ============================================================
# CONVERT FREEHAND LINE INTO CONTINUOUS X-Y CURVE
# ============================================================

# Group points by X.
# For each x, calculate the median Y that you drew.

xmin = int(np.min(line_points[:, 0]))
xmax = int(np.max(line_points[:, 0]))

x_positions = np.arange(
    xmin,
    xmax + 1
)

manual_y = np.full(
    len(x_positions),
    np.nan
)

for i, x in enumerate(x_positions):

    distance = np.abs(
        line_points[:, 0] - x
    )

    nearby = line_points[
        distance <= 2.0
    ]

    if len(nearby) > 0:

        manual_y[i] = np.median(
            nearby[:, 1]
        )

# Fill gaps.
good = np.isfinite(manual_y)

if np.sum(good) < 20:

    cap.release()

    raise RuntimeError(
        "Could not construct continuous manual line."
    )

manual_y = np.interp(
    x_positions,
    x_positions[good],
    manual_y[good]
)

print(
    "Manual line X:",
    xmin,
    "to",
    xmax
)

print(
    "Manual line Y:",
    np.min(manual_y),
    "to",
    np.max(manual_y)
)

# ============================================================
# DISPLAY CONFIRMATION
# ============================================================

check = first.copy()

for x, y in zip(x_positions, manual_y):

    cv2.circle(
        check,
        (int(x), int(y)),
        1,
        (0, 255, 0),
        -1
    )

cv2.imwrite(
    os.path.join(
        OUT,
        "MANUAL_LINE_selection.png"
    ),
    check
)

print(
    "Saved:",
    os.path.join(
        OUT,
        "MANUAL_LINE_selection.png"
    )
)

# ============================================================
# RESET VIDEO
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    TRACKING_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# TRACKING
# ============================================================

all_y = []

frame_number = 0

previous_y = manual_y.copy()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Slight smoothing.
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    current_y = np.zeros_like(
        previous_y
    )

    # --------------------------------------------------------
    # For each x, search ONLY vertically.
    # --------------------------------------------------------

    for i, x in enumerate(x_positions):

        xi = int(x)

        center = int(
            round(previous_y[i])
        )

        y0 = max(
            2,
            center - SEARCH_Y
        )

        y1 = min(
            height - 3,
            center + SEARCH_Y
        )

        # Small vertical strip around this x.
        x0 = max(
            0,
            xi - 2
        )

        x1 = min(
            width,
            xi + 3
        )

        profile = np.mean(
            gray[y0:y1 + 1, x0:x1],
            axis=1
        )

        # Vertical gradient.
        gradient = np.abs(
            np.gradient(profile)
        )

        # Smooth gradient.
        gradient = cv2.GaussianBlur(
            gradient.astype(np.float32),
            (0, 0),
            2
        ).flatten()

        best = np.argmax(
            gradient
        )

        current_y[i] = (
            y0 + best
        )

    # --------------------------------------------------------
    # Smooth the tracked line spatially.
    # --------------------------------------------------------

    current_y = cv2.GaussianBlur(
        current_y.astype(np.float32).reshape(1, -1),
        (0, 0),
        5
    ).flatten()

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    all_y.append(
        current_y.copy()
    )

    previous_y = current_y

    # --------------------------------------------------------
    # Draw tracking line.
    # --------------------------------------------------------

    output = frame.copy()

    pts = np.column_stack(
        [
            x_positions.astype(np.int32),
            current_y.astype(np.int32)
        ]
    )

    cv2.polylines(
        output,
        [pts],
        False,
        (0, 255, 0),
        3
    )

    cv2.putText(
        output,
        "MANUAL LINE - VERTICAL TRACKING",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    writer.write(output)

    frame_number += 1

    if frame_number % 20 == 0:

        print(
            "Processed:",
            frame_number,
            "/",
            nframes
        )

cap.release()
writer.release()

# ============================================================
# TRACKING ARRAY
# ============================================================

tracked_y = np.asarray(
    all_y,
    dtype=np.float64
)

print()
print("Tracking shape:", tracked_y.shape)

# ============================================================
# DISPLACEMENT
# ============================================================

baseline = tracked_y[0]

displacement = (
    tracked_y
    - baseline[None, :]
)

# Common vertical movement.
line_signal = np.median(
    displacement,
    axis=1
)

line_signal -= np.mean(
    line_signal
)

t = np.arange(
    len(line_signal)
) / fps

# ============================================================
# SAVE TRACKING DATA
# ============================================================

np.savez(
    NPZ_OUT,
    tracked_y=tracked_y,
    displacement=displacement,
    x=x_positions,
    baseline=baseline,
    fps=fps,
    row_delay=ROW_DELAY
)

# ============================================================
# PLOT RAW LINE MOTION
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    t,
    line_signal
)

plt.xlabel("Time (s)")
plt.ylabel("Vertical displacement (pixels)")
plt.title("Vertical displacement of manually marked reflection line")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    WAVE_PLOT,
    dpi=200
)

plt.close()

# ============================================================
# REMOVE SLOW MOVEMENT
# ============================================================

# 1 second-ish slow component.
window = max(
    5,
    int(fps * 0.8)
)

kernel = np.ones(window) / window

slow = np.convolve(
    line_signal,
    kernel,
    mode="same"
)

fast_motion = (
    line_signal - slow
)

# Remove edge artifacts from convolution.
fast_motion -= np.mean(
    fast_motion
)

# ============================================================
# PLOT FAST MOTION
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    t,
    fast_motion
)

plt.xlabel("Time (s)")
plt.ylabel("Fast vertical displacement")
plt.title("Fast component of manually marked line motion")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FAST_PLOT,
    dpi=200
)

plt.close()

# ============================================================
# ROLLING SHUTTER
# ============================================================

rs_times = []
rs_values = []

for f in range(len(tracked_y)):

    frame_time = f / fps

    for p in range(len(x_positions)):

        y = tracked_y[f, p]

        value = displacement[f, p]

        capture_time = (
            frame_time
            + y * ROW_DELAY
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

# Remove invalid samples.
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

# ============================================================
# AUDIO RATE
# ============================================================

audio_time = np.arange(
    rs_times[0],
    rs_times[-1],
    1.0 / AUDIO_FS
)

audio = np.interp(
    audio_time,
    rs_times,
    rs_values
)

audio = np.nan_to_num(
    audio
)

audio -= np.mean(audio)

# ============================================================
# BANDPASS
# ============================================================

nyquist = AUDIO_FS / 2

b, a = butter(
    4,
    [20 / nyquist, 250 / nyquist],
    btype="band"
)

audio = filtfilt(
    b,
    a,
    audio
)

audio = np.nan_to_num(
    audio
)

peak = np.max(
    np.abs(audio)
)

print()
print("AUDIO")
print("RMS:", np.sqrt(np.mean(audio ** 2)))
print("Peak:", peak)

if peak > 1e-12:

    audio /= peak

audio *= 0.9

write(
    WAV_OUT,
    AUDIO_FS,
    np.int16(
        np.clip(
            audio,
            -1,
            1
        ) * 32767
    )
)

# ============================================================
# SPECTRUM
# ============================================================

freq, power = welch(
    audio,
    fs=AUDIO_FS,
    nperseg=min(
        65536,
        len(audio)
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
plt.title("Spectrum reconstructed from manually marked line")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    SPECTRUM_PLOT,
    dpi=200
)

plt.close()

print()
print("TOP FREQUENCIES")

idx = np.argsort(
    power[mask]
)[-15:][::-1]

for i in idx:

    print(
        f"{freq[mask][i]:8.2f} Hz   "
        f"{power[mask][i]:.8e}"
    )

print()
print("DONE")
print()
print("Tracking video:")
print(TRACKING_VIDEO)
print()
print("Selection:")
print(
    os.path.join(
        OUT,
        "MANUAL_LINE_selection.png"
    )
)
print()
print("Waveform:")
print(WAVE_PLOT)
print()
print("Fast waveform:")
print(FAST_PLOT)
print()
print("Spectrum:")
print(SPECTRUM_PLOT)
print()
print("WAV:")
print(WAV_OUT)