# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

# ============================================================
# PATHS
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4"

OUT_DIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

os.makedirs(OUT_DIR, exist_ok=True)

TRACK_NPY = os.path.join(
    OUT_DIR,
    "marked_line_tracking.npy"
)

TRACK_VIDEO = os.path.join(
    OUT_DIR,
    "marked_line_tracking.mp4"
)

WAV_OUT = os.path.join(
    OUT_DIR,
    "marked_line_RS_bass.wav"
)

# ============================================================
# CAMERA / VIDEO
# ============================================================

RS_DELAY = 7.7435e-6
OUT_SR = 48000

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video:", VIDEO)
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Frames:", nframes)

# ============================================================
# READ FIRST FRAME
# ============================================================

ok, first = cap.read()

if not ok:
    raise RuntimeError("Could not read first frame.")

display = first.copy()

instructions = [
    "CLICK 5-15 POINTS ALONG THE ACTUAL WHITE REFLECTION LINE",
    "Click from LEFT to RIGHT.",
    "Do NOT click the water above/below it.",
    "Press ENTER when finished.",
    "Press R to reset points.",
    "Press ESC to cancel."
]

for i, txt in enumerate(instructions):
    cv2.putText(
        display,
        txt,
        (20, 35 + i * 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

clicked = []

# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        clicked.append(
            (float(x), float(y))
        )

        cv2.circle(
            display,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.imshow(
            "MARK THE LINE",
            display
        )

# ============================================================
# MANUAL LINE SELECTION
# ============================================================

cv2.namedWindow(
    "MARK THE LINE",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "MARK THE LINE",
    1100,
    900
)

cv2.setMouseCallback(
    "MARK THE LINE",
    mouse_callback
)

cv2.imshow(
    "MARK THE LINE",
    display
)

print()
print("==========================================")
print("MANUAL LINE SELECTION")
print("==========================================")
print("Click 5-15 points along the WHOLE WHITE LINE.")
print("Go from left to right.")
print("Press ENTER when finished.")
print("Press R to reset.")
print("Press ESC to cancel.")
print("==========================================")

while True:

    key = cv2.waitKey(20) & 0xFF

    if key == 13:
        break

    if key == ord("r"):

        clicked.clear()

        display = first.copy()

        for i, txt in enumerate(instructions):

            cv2.putText(
                display,
                txt,
                (20, 35 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow(
            "MARK THE LINE",
            display
        )

    if key == 27:

        cv2.destroyAllWindows()
        cap.release()
        raise SystemExit

cv2.destroyAllWindows()

if len(clicked) < 3:

    cap.release()

    raise RuntimeError(
        "Need at least 3 points."
    )

# ============================================================
# SORT MANUAL POINTS LEFT -> RIGHT
# ============================================================

clicked = sorted(
    clicked,
    key=lambda p: p[0]
)

click_x = np.array(
    [p[0] for p in clicked],
    dtype=float
)

click_y = np.array(
    [p[1] for p in clicked],
    dtype=float
)

print("\nSelected points:")

for x, y in zip(click_x, click_y):
    print(
        f"x={x:.1f}, y={y:.1f}"
    )

# ============================================================
# DENSE LINE
# ============================================================

x_min = int(np.ceil(click_x.min()))
x_max = int(np.floor(click_x.max()))

# Do not use every pixel.
# 300-500 spatial samples are enough.
n_x = min(
    500,
    max(100, x_max - x_min + 1)
)

line_x = np.linspace(
    x_min,
    x_max,
    n_x
)

interp = interp1d(
    click_x,
    click_y,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)

line_y0 = interp(line_x)

print("\nDense line:")
print("X range:", x_min, "-", x_max)
print("Samples:", n_x)
print(
    "Initial Y range:",
    line_y0.min(),
    "-",
    line_y0.max()
)

# ============================================================
# TRACKING ARRAYS
# ============================================================

tracked_y = np.full(
    (nframes, n_x),
    np.nan,
    dtype=np.float32
)

tracked_y[0] = line_y0

# ============================================================
# TRACK FUNCTION
# ============================================================

def track_line(frame, previous_y):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Smooth enough to suppress pixel noise
    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        2.0
    )

    # Vertical gradient
    gy = cv2.Sobel(
        smooth,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    result = np.zeros(
        n_x,
        dtype=np.float32
    )

    for i, xx in enumerate(line_x):

        xi = int(round(xx))

        if xi < 3 or xi >= width - 3:
            result[i] = previous_y[i]
            continue

        center = int(round(previous_y[i]))

        lo = max(
            3,
            center - 14
        )

        hi = min(
            height - 4,
            center + 15
        )

        ys = np.arange(
            lo,
            hi
        )

        # ----------------------------------------------------
        # Bright-line ridge score
        #
        # narrow blur = local bright structure
        # broad blur  = surrounding background
        # ----------------------------------------------------

        profile = smooth[
            ys,
            xi
        ].astype(float)

        # Local background estimate
        bg = cv2.GaussianBlur(
            profile.reshape(-1, 1).astype(np.float32),
            (0, 0),
            4
        ).ravel()

        score = profile - bg

        # Also use local vertical gradient
        grad = np.abs(
            gy[
                ys,
                xi
            ]
        )

        grad /= (
            np.max(grad) + 1e-9
        )

        score_norm = score - np.min(score)

        score_norm /= (
            np.max(score_norm) + 1e-9
        )

        combined = (
            0.65 * score_norm
            +
            0.35 * grad
        )

        best = np.argmax(
            combined
        )

        result[i] = ys[best]

    # ========================================================
    # SPATIAL SMOOTHING
    # ========================================================

    result = cv2.GaussianBlur(
        result.reshape(1, -1),
        (0, 0),
        2.0
    ).ravel()

    return result

# ============================================================
# VIDEO OUTPUT
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    TRACK_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# TRACK VIDEO
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

previous_y = line_y0.copy()

for frame_idx in range(nframes):

    ok, frame = cap.read()

    if not ok:
        break

    if frame_idx == 0:

        current_y = line_y0.copy()

    else:

        current_y = track_line(
            frame,
            previous_y
        )

    tracked_y[
        frame_idx
    ] = current_y

    previous_y = current_y

    # --------------------------------------------------------
    # DRAW TRACKED LINE
    # --------------------------------------------------------

    overlay = frame.copy()

    pts = np.column_stack(
        (
            line_x,
            current_y
        )
    ).astype(
        np.int32
    )

    cv2.polylines(
        overlay,
        [pts.reshape(-1, 1, 2)],
        False,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Draw original selected points
    for xx, yy in zip(
        click_x,
        click_y
    ):

        cv2.circle(
            overlay,
            (
                int(xx),
                int(yy)
            ),
            4,
            (0, 0, 255),
            -1
        )

    cv2.putText(
        overlay,
        f"Frame {frame_idx+1}/{nframes}",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    writer.write(
        overlay
    )

    if frame_idx % 30 == 0:

        print(
            f"Tracking frame "
            f"{frame_idx+1}/{nframes}"
        )

writer.release()
cap.release()

# ============================================================
# REMOVE BAD TRACKING VALUES
# ============================================================

for j in range(n_x):

    col = tracked_y[:, j]

    good = np.isfinite(col)

    if np.sum(good) >= 2:

        tracked_y[:, j] = np.interp(
            np.arange(nframes),
            np.flatnonzero(good),
            col[good]
        )

# ============================================================
# BASELINE
# ============================================================

baseline = np.median(
    tracked_y,
    axis=0
)

# Vertical deformation
deformation = (
    tracked_y
    -
    baseline[None, :]
)

# Remove slow spatial baseline variation
deformation -= np.median(
    deformation,
    axis=1,
    keepdims=True
)

# ============================================================
# SAVE TRACKING DATA
# ============================================================

np.savez(
    TRACK_NPY.replace(
        ".npy",
        ".npz"
    ),
    tracked_y=tracked_y,
    deformation=deformation,
    baseline=baseline,
    x=line_x,
    fps=fps,
    row_delay=RS_DELAY
)

# ============================================================
# ROLLING-SHUTTER RECONSTRUCTION
# ============================================================

print()
print("Building rolling-shutter signal...")

samples_t = []
samples_v = []

for f in range(nframes):

    frame_time = (
        f / fps
    )

    ys = tracked_y[f]
    ds = deformation[f]

    # Each row is captured at a slightly
    # different time.
    times = (
        frame_time
        +
        ys * RS_DELAY
    )

    for k in range(n_x):

        if np.isfinite(
            times[k]
        ) and np.isfinite(
            ds[k]
        ):

            samples_t.append(
                times[k]
            )

            samples_v.append(
                ds[k]
            )

samples_t = np.asarray(
    samples_t,
    dtype=float
)

samples_v = np.asarray(
    samples_v,
    dtype=float
)

# Sort by rolling-shutter time
order = np.argsort(
    samples_t
)

samples_t = samples_t[order]
samples_v = samples_v[order]

# Remove duplicate / extremely close timestamps
dt = np.diff(
    samples_t
)

keep = np.concatenate(
    (
        [True],
        dt > 1e-8
    )
)

samples_t = samples_t[keep]
samples_v = samples_v[keep]

# ============================================================
# RESAMPLE TO 48 kHz
# ============================================================

start = samples_t.min()
end = samples_t.max()

audio_t = np.arange(
    start,
    end,
    1.0 / OUT_SR
)

audio = np.interp(
    audio_t,
    samples_t,
    samples_v
)

audio -= np.mean(
    audio
)

# ============================================================
# BANDPASS
# ============================================================

low = 20.0
high = min(
    250.0,
    OUT_SR / 2 - 1
)

b, a = signal.butter(
    4,
    [
        low / (OUT_SR / 2),
        high / (OUT_SR / 2)
    ],
    btype="band"
)

audio = signal.filtfilt(
    b,
    a,
    audio
)

# ============================================================
# REMOVE EXTREME OUTLIERS
# ============================================================

limit = 5.0 * np.std(
    audio
)

audio = np.clip(
    audio,
    -limit,
    limit
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

# Moderate gain only
audio *= 0.85

wavfile.write(
    WAV_OUT,
    OUT_SR,
    np.int16(
        audio * 32767
    )
)

# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

import matplotlib.pyplot as plt

# Mean/median line deformation
line_signal = np.median(
    deformation,
    axis=1
)

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    np.arange(nframes) / fps,
    line_signal
)

plt.xlabel("Time (s)")
plt.ylabel("Vertical deformation")
plt.title("Marked-Line Vertical Deformation")
plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "marked_line_deformation.png"
    ),
    dpi=160
)

plt.close()

# ============================================================
# DONE
# ============================================================

print()
print("==========================================")
print("MARKED-LINE TRACKING COMPLETE")
print("==========================================")
print("Tracking video:")
print(TRACK_VIDEO)
print()
print("Tracking data:")
print(
    TRACK_NPY.replace(
        ".npy",
        ".npz"
    )
)
print()
print("Recovered WAV:")
print(WAV_OUT)
print()
print("Diagnostic:")
print(
    os.path.join(
        OUT_DIR,
        "marked_line_deformation.png"
    )
)
print("==========================================")