import cv2
import numpy as np
import pyrtools as pt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\itsybitsy.mp4"

OUT_DIR = Path(
    r"C:\Users\Harkiran\Downloads"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Experimental rolling-shutter calibration
ROW_DELAY = 7.7435e-6

AUDIO_FS = 48000

# Existing ROI
X1, X2 = 162, 918
Y1, Y2 = 192, 1728

# 4 x 2 = 8 spatial patches
PATCH_COLS = 4
PATCH_ROWS = 2

# Keep pyramid manageable
PYRAMID_HEIGHT = 3

# Speech band
SPEECH_LOW = 80
SPEECH_HIGH = 5000


# ============================================================
# FUNCTIONS
# ============================================================

def bandpass(x, fs, low, high, order=4):

    high = min(
        high,
        fs * 0.45
    )

    sos = signal.butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return signal.sosfiltfilt(
        sos,
        x
    )


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


def save_wav(path, x):

    x = normalize(x)

    pcm = np.int16(
        np.clip(
            x * 0.9,
            -1,
            1
        ) * 32767
    )

    wavfile.write(
        str(path),
        AUDIO_FS,
        pcm
    )


def get_complex_bands(pyr):

    bands = []

    for key, coeff in pyr.pyr_coeffs.items():

        if not np.iscomplexobj(coeff):
            continue

        h, w = coeff.shape

        if h < 8 or w < 8:
            continue

        bands.append(
            (key, coeff)
        )

    return bands


def ecc_stabilize(reference, frame):

    ref = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    cur = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    ref = cv2.GaussianBlur(
        ref,
        (5, 5),
        0
    )

    cur = cv2.GaussianBlur(
        cur,
        (5, 5),
        0
    )

    warp = np.eye(
        2,
        3,
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        1e-5
    )

    try:

        _, warp = cv2.findTransformECC(
            ref,
            cur,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            1
        )

        return cv2.warpAffine(
            frame,
            warp,
            (
                frame.shape[1],
                frame.shape[0]
            ),
            flags=cv2.INTER_LINEAR |
                  cv2.WARP_INVERSE_MAP |
                  cv2.WARP_FILL_OUTLIERS
        )

    except cv2.error:

        return frame


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open:\n{VIDEO}"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_count = int(
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
print("=" * 60)
print("SPATIAL-PATCH VISUAL MICROPHONE")
print("=" * 60)
print(f"Video      : {VIDEO}")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.6f}")
print(f"Frames     : {frame_count}")
print(f"RS delay   : {ROW_DELAY:.9e} s/row")
print(f"ROI        : x={X1}:{X2}, y={Y1}:{Y2}")
print(
    f"Patches    : "
    f"{PATCH_COLS} x {PATCH_ROWS} = "
    f"{PATCH_COLS * PATCH_ROWS}"
)
print("=" * 60)
print()


# ============================================================
# FIRST FRAME
# ============================================================

ok, first_frame = cap.read()

if not ok:
    raise RuntimeError(
        "Could not read first frame."
    )

reference = first_frame.copy()


# ============================================================
# PATCH GEOMETRY
# ============================================================

roi_width = X2 - X1
roi_height = Y2 - Y1

patch_width = roi_width // PATCH_COLS
patch_height = roi_height // PATCH_ROWS

print(
    f"Patch size: "
    f"{patch_width} x {patch_height}"
)
print()


# ============================================================
# BUILD REFERENCE PYRAMIDS FOR EACH PATCH
# ============================================================

patches = []

gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)

for py in range(PATCH_ROWS):

    for px in range(PATCH_COLS):

        px1 = X1 + px * patch_width
        px2 = (
            X1 +
            (px + 1) * patch_width
        )

        py1 = Y1 + py * patch_height
        py2 = (
            Y1 +
            (py + 1) * patch_height
        )

        roi = gray[
            py1:py2,
            px1:px2
        ]

        roi = (
            roi.astype(
                np.float32
            ) / 255.0
        )

        print(
            f"Building patch "
            f"{len(patches):02d}: "
            f"x={px1}:{px2}, "
            f"y={py1}:{py2}"
        )

        pyr = pt.pyramids.SteerablePyramidFreq(
            roi,
            height=PYRAMID_HEIGHT,
            order=3,
            is_complex=True
        )

        bands = get_complex_bands(
            pyr
        )

        patches.append({
            "id": len(patches),
            "px": px,
            "py": py,
            "x1": px1,
            "x2": px2,
            "y1": py1,
            "y2": py2,
            "width": px2 - px1,
            "height": py2 - py1,
            "bands": [
                {
                    "key": key,
                    "previous": coeff.copy(),
                    "rows": []
                }
                for key, coeff in bands
            ]
        })


print()
print(
    f"Total patches: {len(patches)}"
)
print()


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_index = 1

while True:

    ok, frame = cap.read()

    if not ok:
        break

    # --------------------------------------------------------
    # Global camera stabilization
    # --------------------------------------------------------

    stabilized = ecc_stabilize(
        reference,
        frame
    )

    gray = cv2.cvtColor(
        stabilized,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Process every spatial patch
    # --------------------------------------------------------

    for patch in patches:

        roi = gray[
            patch["y1"]:patch["y2"],
            patch["x1"]:patch["x2"]
        ]

        roi = (
            roi.astype(
                np.float32
            ) / 255.0
        )

        pyr = pt.pyramids.SteerablePyramidFreq(
            roi,
            height=PYRAMID_HEIGHT,
            order=3,
            is_complex=True
        )

        current_bands = dict(
            get_complex_bands(pyr)
        )

        for band in patch["bands"]:

            key = band["key"]

            if key not in current_bands:
                continue

            current = current_bands[key]
            previous = band["previous"]

            # ------------------------------------------------
            # Complex phase difference
            # ------------------------------------------------

            phase = np.angle(
                current *
                np.conj(previous)
            )

            # ------------------------------------------------
            # Amplitude weighting
            # ------------------------------------------------

            amplitude = (
                np.abs(current) +
                np.abs(previous)
            ) * 0.5

            weight = amplitude ** 2

            weight /= (
                np.mean(weight)
                + 1e-8
            )

            weight = np.clip(
                weight,
                0,
                20
            )

            weighted_phase = (
                phase * weight
            )

            # ------------------------------------------------
            # Resize to patch row dimension
            # ------------------------------------------------

            resized = cv2.resize(
                weighted_phase.astype(
                    np.float32
                ),
                (
                    patch["width"],
                    patch["height"]
                ),
                interpolation=cv2.INTER_LINEAR
            )

            # ------------------------------------------------
            # Average horizontally ONLY INSIDE THIS PATCH
            # ------------------------------------------------

            row_signal = np.mean(
                resized,
                axis=1
            )

            band["rows"].append(
                row_signal
            )

            band["previous"] = (
                current.copy()
            )

    if frame_index % 25 == 0:

        print(
            f"Processed "
            f"{frame_index:4d} / "
            f"{frame_count}"
        )

    frame_index += 1


cap.release()

print()
print("Video processing complete.")
print()


# ============================================================
# RECONSTRUCT EACH PATCH
# ============================================================

patch_candidates = []

print("=" * 60)
print("RECONSTRUCTING PATCH SIGNALS")
print("=" * 60)
print()


for patch in patches:

    patch_signals = []

    for band_index, band in enumerate(
        patch["bands"]
    ):

        if len(band["rows"]) < 10:
            continue

        matrix = np.asarray(
            band["rows"],
            dtype=np.float64
        )

        n_frames, n_rows = (
            matrix.shape
        )

        # Remove frame-wise spatial DC
        matrix -= np.mean(
            matrix,
            axis=1,
            keepdims=True
        )

        # Remove row-wise temporal DC
        matrix -= np.mean(
            matrix,
            axis=0,
            keepdims=True
        )

        # ----------------------------------------------------
        # Rolling-shutter timestamps
        # ----------------------------------------------------

        frame_numbers = np.repeat(
            np.arange(n_frames),
            n_rows
        )

        row_numbers = np.tile(
            np.arange(n_rows),
            n_frames
        )

        times = (
            frame_numbers / fps
            +
            row_numbers * ROW_DELAY
        )

        values = matrix.ravel()

        order = np.argsort(
            times
        )

        times = times[order]
        values = values[order]

        keep = np.concatenate(
            (
                [True],
                np.diff(times) > 0
            )
        )

        times = times[keep]
        values = values[keep]

        # ----------------------------------------------------
        # Regular audio grid
        # ----------------------------------------------------

        audio_time = np.arange(
            times[0],
            times[-1],
            1.0 / AUDIO_FS
        )

        interpolator = interp1d(
            times,
            values,
            kind="linear",
            bounds_error=False,
            fill_value=0
        )

        audio = interpolator(
            audio_time
        )

        # ----------------------------------------------------
        # Speech-band filtering
        # ----------------------------------------------------

        audio = bandpass(
            audio,
            AUDIO_FS,
            SPEECH_LOW,
            SPEECH_HIGH
        )

        audio = normalize(
            audio
        )

        patch_signals.append({
            "patch": patch["id"],
            "px": patch["px"],
            "py": patch["py"],
            "band": band_index,
            "key": band["key"],
            "audio": audio
        })

    # --------------------------------------------------------
    # Save the strongest signal from every patch
    # --------------------------------------------------------

    if len(patch_signals) == 0:
        continue

    # Score based on speech-band energy versus very-low
    # frequency contamination.

    scored = []

    for candidate in patch_signals:

        x = candidate["audio"]

        f, pxx = signal.welch(
            x,
            fs=AUDIO_FS,
            nperseg=min(
                16384,
                len(x)
            )
        )

        speech = (
            (f >= 250) &
            (f <= 4000)
        )

        low = (
            (f >= 20) &
            (f < 250)
        )

        speech_energy = np.sum(
            pxx[speech]
        )

        low_energy = (
            np.sum(
                pxx[low]
            )
            + 1e-12
        )

        score = (
            speech_energy /
            low_energy
        )

        candidate["score"] = score

        scored.append(
            candidate
        )

    scored.sort(
        key=lambda z: z["score"],
        reverse=True
    )

    best = scored[0]

    patch_candidates.append(
        best
    )

    output = (
        OUT_DIR /
        f"patch_{patch['id']:02d}"
        f"_best_band_{best['band']:02d}.wav"
    )

    save_wav(
        output,
        best["audio"]
    )

    print(
        f"Patch {patch['id']:02d} "
        f"({patch['px']},{patch['py']}) "
        f"best band={best['band']:02d} "
        f"score={best['score']:.4e}"
    )

    print(
        f"    saved: {output.name}"
    )


# ============================================================
# SAVE PATCH MAP
# ============================================================

print()
print("=" * 60)
print("PATCH MAP")
print("=" * 60)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

for patch in patches:

    cx = (
        patch["x1"] +
        patch["x2"]
    ) / 2

    cy = (
        patch["y1"] +
        patch["y2"]
    ) / 2

    ax.add_patch(
        plt.Rectangle(
            (
                patch["x1"],
                patch["y1"]
            ),
            patch["width"],
            patch["height"],
            fill=False
        )
    )

    ax.text(
        cx,
        cy,
        f"P{patch['id']}",
        ha="center",
        va="center",
        fontsize=14
    )

ax.set_xlim(
    X1,
    X2
)

ax.set_ylim(
    Y2,
    Y1
)

ax.set_title(
    "Spatial Patch Layout"
)

ax.set_xlabel(
    "Image X"
)

ax.set_ylabel(
    "Image Y"
)

patch_map = (
    OUT_DIR /
    "patch_layout.png"
)

plt.tight_layout()

plt.savefig(
    patch_map,
    dpi=150
)

plt.close()

print(
    f"Saved: {patch_map}"
)


# ============================================================
# BUILD PATCH COMBINATION
# ============================================================

print()
print("=" * 60)
print("BUILDING PATCH COMBINATION")
print("=" * 60)

if len(patch_candidates) == 0:

    raise RuntimeError(
        "No patch candidates were generated."
    )

length = min(
    len(x["audio"])
    for x in patch_candidates
)

combined = np.zeros(
    length,
    dtype=np.float64
)

# Normalize contribution of each patch
for candidate in patch_candidates:

    x = candidate[
        "audio"
    ][:length]

    combined += (
        x /
        max(
            len(patch_candidates),
            1
        )
    )

combined = normalize(
    combined
)

combined_path = (
    OUT_DIR /
    "patch_combined.wav"
)

save_wav(
    combined_path,
    combined
)

print(
    f"Saved: {combined_path}"
)


# ============================================================
# SPECTROGRAM
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.specgram(
    combined,
    NFFT=2048,
    Fs=AUDIO_FS,
    noverlap=1536
)

plt.ylim(
    0,
    5000
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Frequency (Hz)"
)

plt.title(
    "Spatial-Patch Visual Microphone Reconstruction"
)

plt.colorbar(
    label="Power"
)

spectrogram_path = (
    OUT_DIR /
    "patch_combined_spectrogram.png"
)

plt.tight_layout()

plt.savefig(
    spectrogram_path,
    dpi=150
)

plt.close()

print(
    f"Saved: {spectrogram_path}"
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print()
print(
    f"Output folder:\n{OUT_DIR}"
)
print()
print(
    "Listen to patch_00 through patch_07"
)
print(
    "and then patch_combined.wav"
)
print()