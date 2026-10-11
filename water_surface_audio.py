import cv2
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

VIDEO = Path(
    r"C:\Users\Harkiran\Downloads\Mobile Devices\itsybitsy.mp4"
)

OUT_DIR = Path(
    r"C:\Users\Harkiran\OneDrive\visual mic"
) / "water_surface"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Existing water ROI
X1, X2 = 162, 918
Y1, Y2 = 192, 1728

# Calibrated rolling shutter delay
ROW_DELAY = 7.7435e-6

AUDIO_FS = 48000

# Speech band
LOW = 80
HIGH = 5000

# How many pixels around the detected surface to analyze
SURFACE_HALF_WIDTH = 25

# Number of x-columns to sample
NUM_X = 120


# ============================================================
# HELPERS
# ============================================================

def normalize(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    x -= np.mean(x)

    peak = np.max(
        np.abs(x)
    )

    if peak > 1e-12:
        x /= peak

    return x


def bandpass(x, fs, low, high):

    high = min(
        high,
        fs * 0.45
    )

    sos = signal.butter(
        4,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return signal.sosfiltfilt(
        sos,
        x
    )


def save_wav(path, x):

    x = normalize(x)

    wavfile.write(
        str(path),
        AUDIO_FS,
        np.int16(
            np.clip(
                x * 0.9,
                -1,
                1
            ) * 32767
        )
    )


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO)
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open:\n{VIDEO}"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

print()
print("=" * 70)
print("WATER-SURFACE VISUAL MICROPHONE")
print("=" * 70)
print(
    f"Video      : {VIDEO}"
)
print(
    f"Resolution : {width} x {height}"
)
print(
    f"FPS        : {fps:.6f}"
)
print(
    f"Frames     : {frames}"
)
print(
    f"ROI        : x={X1}:{X2}, y={Y1}:{Y2}"
)
print(
    f"RS delay   : {ROW_DELAY:.9e} s/row"
)
print("=" * 70)
print()


# ============================================================
# FIRST FRAME
# ============================================================

ok, first = cap.read()

if not ok:

    raise RuntimeError(
        "Could not read first frame."
    )

first_gray = cv2.cvtColor(
    first,
    cv2.COLOR_BGR2GRAY
)

first_roi = first_gray[
    Y1:Y2,
    X1:X2
]

# Slight blur to reduce compression noise
first_blur = cv2.GaussianBlur(
    first_roi,
    (5, 5),
    0
)

# Vertical gradient = horizontal surface/edge
gradient = cv2.Sobel(
    first_blur,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

gradient_abs = np.abs(
    gradient
)

# Average gradient over x.
# A water surface should create a strong horizontal structure.
profile = np.mean(
    gradient_abs,
    axis=1
)

# Ignore ROI boundaries
margin = 40

search_profile = profile.copy()

search_profile[:margin] = 0
search_profile[-margin:] = 0

surface_y = int(
    np.argmax(
        search_profile
    )
)

print(
    f"Detected water-surface row "
    f"inside ROI: {surface_y}"
)

surface_global_y = (
    Y1 +
    surface_y
)

print(
    f"Global image row: "
    f"{surface_global_y}"
)

# ============================================================
# SAVE DIAGNOSTIC FRAME
# ============================================================

diagnostic = first.copy()

cv2.line(
    diagnostic,
    (X1, surface_global_y),
    (X2, surface_global_y),
    (0, 0, 255),
    3
)

cv2.rectangle(
    diagnostic,
    (
        X1,
        max(
            Y1,
            surface_global_y -
            SURFACE_HALF_WIDTH
        )
    ),
    (
        X2,
        min(
            Y2,
            surface_global_y +
            SURFACE_HALF_WIDTH
        )
    ),
    (255, 0, 0),
    2
)

diagnostic_path = (
    OUT_DIR /
    "01_detected_water_surface.png"
)

cv2.imwrite(
    str(diagnostic_path),
    diagnostic
)

print(
    f"Saved diagnostic: "
    f"{diagnostic_path}"
)

# ============================================================
# X POSITIONS
# ============================================================

xs = np.linspace(
    0,
    X2 - X1 - 1,
    NUM_X
).astype(int)

# Reference intensity profiles around surface
reference_profiles = []

for x in xs:

    xa = max(
        0,
        x - 3
    )

    xb = min(
        first_roi.shape[1],
        x + 4
    )

    column = np.mean(
        first_blur[
            :,
            xa:xb
        ],
        axis=1
    )

    reference_profiles.append(
        column
    )

reference_profiles = np.asarray(
    reference_profiles
)

# ============================================================
# FRAME PROCESSING
# ============================================================

# For each frame:
#
#   estimate the local water surface position
#   at multiple x locations.
#
# This gives a spatially distributed displacement
# measurement instead of one global phase signal.

surface_positions = []

frame_number = 0

# Rewind
cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

while True:

    ok, frame = cap.read()

    if not ok:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[
        Y1:Y2,
        X1:X2
    ]

    roi = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    grad = cv2.Sobel(
        roi,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    grad_abs = np.abs(
        grad
    )

    positions = []

    # Search locally around reference surface.
    search_min = max(
        margin,
        surface_y - 120
    )

    search_max = min(
        roi.shape[0] - margin,
        surface_y + 120
    )

    for x in xs:

        xa = max(
            0,
            x - 3
        )

        xb = min(
            roi.shape[1],
            x + 4
        )

        local = np.mean(
            grad_abs[
                search_min:search_max,
                xa:xb
            ],
            axis=1
        )

        if len(local) == 0:

            positions.append(
                np.nan
            )

            continue

        local_y = (
            search_min +
            np.argmax(local)
        )

        positions.append(
            local_y
        )

    surface_positions.append(
        positions
    )

    frame_number += 1

    if frame_number % 25 == 0:

        print(
            f"Processed "
            f"{frame_number:4d} / "
            f"{frames}"
        )


cap.release()

surface_positions = np.asarray(
    surface_positions,
    dtype=np.float64
)

print()
print(
    "Surface tracking complete."
)

print(
    "Tracking matrix:",
    surface_positions.shape
)


# ============================================================
# INTERPOLATE MISSING POSITIONS
# ============================================================

for j in range(
    surface_positions.shape[1]
):

    y = surface_positions[:, j]

    good = np.isfinite(y)

    if np.sum(good) < 10:

        continue

    bad = ~good

    if np.any(bad):

        surface_positions[
            bad,
            j
        ] = np.interp(
            np.flatnonzero(bad),
            np.flatnonzero(good),
            y[good]
        )


# ============================================================
# REMOVE SLOW DRIFT
# ============================================================

# displacement relative to temporal median

displacement = (
    surface_positions -
    np.median(
        surface_positions,
        axis=0,
        keepdims=True
    )
)

# Remove very slow movement
for j in range(
    displacement.shape[1]
):

    displacement[:, j] = (
        signal.detrend(
            displacement[:, j]
        )
    )


# ============================================================
# SAVE TRACKING MATRIX
# ============================================================

np.save(
    OUT_DIR /
    "surface_displacement.npy",
    displacement
)

# ============================================================
# SURFACE TRACKING VISUALIZATION
# ============================================================

plt.figure(
    figsize=(13, 6)
)

plt.imshow(
    displacement.T,
    aspect="auto",
    origin="lower",
    extent=[
        0,
        len(displacement) / fps,
        0,
        NUM_X
    ]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Surface position sample"
)

plt.title(
    "Tracked Water-Surface Displacement"
)

plt.colorbar(
    label="Displacement (pixels)"
)

plt.tight_layout()

tracking_plot = (
    OUT_DIR /
    "02_surface_displacement.png"
)

plt.savefig(
    tracking_plot,
    dpi=160
)

plt.close()

print(
    f"Saved: {tracking_plot}"
)


# ============================================================
# CANDIDATE 1:
# SPATIAL MEDIAN
# ============================================================

median_motion = np.median(
    displacement,
    axis=1
)

# Frame-rate signal first
median_motion = normalize(
    median_motion
)


# ============================================================
# CANDIDATE 2:
# SPATIAL PCA
# ============================================================

M = displacement.copy()

M -= np.mean(
    M,
    axis=0,
    keepdims=True
)

U, S, Vt = np.linalg.svd(
    M,
    full_matrices=False
)

pca1 = U[:, 0] * S[0]
pca2 = U[:, 1] * S[1]
pca3 = U[:, 2] * S[2]

pca1 = normalize(pca1)
pca2 = normalize(pca2)
pca3 = normalize(pca3)


# ============================================================
# ROLLING-SHUTTER RECONSTRUCTION
# ============================================================

def reconstruct_rs(
    signal_matrix,
    name
):

    n_frames, n_x = (
        signal_matrix.shape
    )

    # IMPORTANT:
    #
    # We don't assume every x position corresponds
    # to a different rolling-shutter time.
    #
    # Instead, use the tracked surface positions
    # as a spatial vibration field and preserve
    # the vertical location of the surface.

    # Build a row-index signal from the water surface.
    #
    # Each tracked displacement is assigned to the
    # row where that surface was observed.

    values = []
    times = []

    for f in range(
        n_frames
    ):

        for j, x in enumerate(xs):

            row = (
                surface_positions[f, j]
            )

            if not np.isfinite(row):
                continue

            t = (
                f / fps +
                row * ROW_DELAY
            )

            values.append(
                signal_matrix[f, j]
            )

            times.append(
                t
            )

    values = np.asarray(
        values,
        dtype=np.float64
    )

    times = np.asarray(
        times,
        dtype=np.float64
    )

    order = np.argsort(
        times
    )

    times = times[order]
    values = values[order]

    good = np.concatenate(
        (
            [True],
            np.diff(times) > 0
        )
    )

    times = times[good]
    values = values[good]

    if len(times) < 100:

        return None

    audio_time = np.arange(
        times[0],
        times[-1],
        1.0 / AUDIO_FS
    )

    interp = interp1d(
        times,
        values,
        kind="linear",
        bounds_error=False,
        fill_value=0
    )

    audio = interp(
        audio_time
    )

    audio = bandpass(
        audio,
        AUDIO_FS,
        LOW,
        HIGH
    )

    audio = normalize(
        audio
    )

    path = (
        OUT_DIR /
        f"{name}.wav"
    )

    save_wav(
        path,
        audio
    )

    print(
        f"Saved: {path.name}"
    )

    return audio


# ============================================================
# IMPORTANT:
# PCA signals are frame-domain signals, so for RS
# reconstruction we need a spatial field.
#
# Therefore first reconstruct from actual displacement.
# ============================================================

# Candidate A:
# spatial median displacement replicated across x
field_median = np.repeat(
    median_motion[:, None],
    NUM_X,
    axis=1
)

# Candidate B:
# use each individual surface displacement
field_actual = displacement.copy()


# ============================================================
# RECONSTRUCT
# ============================================================

audio_actual = reconstruct_rs(
    field_actual,
    "water_surface_rs_actual"
)

audio_median = reconstruct_rs(
    field_median,
    "water_surface_rs_median"
)


# ============================================================
# PCA AUDIO — FRAME DOMAIN
# ============================================================

def frame_audio(
    x,
    name
):

    x = bandpass(
        x,
        fps,
        1.0,
        min(
            14.0,
            fps * 0.45
        )
    )

    x = normalize(
        x
    )

    # Resample only for listening.
    target_time = np.arange(
        0,
        len(x) / fps,
        1 / AUDIO_FS
    )

    source_time = np.arange(
        len(x)
    ) / fps

    y = np.interp(
        target_time,
        source_time,
        x
    )

    path = (
        OUT_DIR /
        f"{name}.wav"
    )

    save_wav(
        path,
        y
    )

    print(
        f"Saved: {path.name}"
    )


frame_audio(
    pca1,
    "water_surface_pca1"
)

frame_audio(
    pca2,
    "water_surface_pca2"
)

frame_audio(
    pca3,
    "water_surface_pca3"
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print()
print(
    f"Output folder:\n{OUT_DIR}"
)
print()
print(
    "FIRST CHECK:"
)
print(
    "  01_detected_water_surface.png"
)
print()
print(
    "THEN LISTEN:"
)
print(
    "  water_surface_rs_actual.wav"
)
print(
    "  water_surface_rs_median.wav"
)
print(
    "  water_surface_pca1.wav"
)
print(
    "  water_surface_pca2.wav"
)
print(
    "  water_surface_pca3.wav"
)
print()