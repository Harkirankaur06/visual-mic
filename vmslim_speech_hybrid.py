import cv2
import numpy as np
import pyrtools as pt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\itsybitsy.mp4"

OUT_DIR = Path(r"C:\Users\Harkiran\OneDrive\visual mic\speech_hybrid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Experimental rolling-shutter delay obtained from A15 calibration
ROW_DELAY = 7.7435e-6

# Output audio
AUDIO_FS = 48000

# Speech band
LOW_HZ = 80
HIGH_HZ = 5000

# ROI used successfully in the previous experiment
X1, X2 = 162, 918
Y1, Y2 = 192, 1728

# Limit pyramid height to avoid excessive computation
PYRAMID_HEIGHT = 4

# ============================================================
# HELPERS
# ============================================================

def bandpass(x, fs, low=80, high=5000, order=5):
    high = min(high, fs * 0.45)

    sos = signal.butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return signal.sosfiltfilt(sos, x)


def normalize_audio(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    peak = np.max(np.abs(x))

    if peak > 1e-12:
        x = x / peak

    return x


def save_wav(path, x, fs=AUDIO_FS):
    x = normalize_audio(x)

    # Leave some headroom
    x = 0.90 * x

    pcm = np.int16(np.clip(x, -1, 1) * 32767)

    wavfile.write(str(path), fs, pcm)


def ecc_stabilize(reference, frame):
    """
    Estimate global affine motion relative to reference.
    Only used to remove large camera movement.
    """

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

    warp = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5
    )

    try:
        _, warp = cv2.findTransformECC(
            ref_gray,
            cur_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            1
        )

        stabilized = cv2.warpAffine(
            frame,
            warp,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR |
                  cv2.WARP_INVERSE_MAP |
                  cv2.WARP_FILL_OUTLIERS
        )

        return stabilized

    except cv2.error:
        # If ECC fails, keep the frame rather than stopping.
        return frame


def get_complex_bands(pyr):
    """
    Return complex pyramid bands only.
    """
    result = []

    for key, coeff in pyr.pyr_coeffs.items():

        if not np.iscomplexobj(coeff):
            continue

        h, w = coeff.shape

        # Ignore extremely tiny bands
        if h < 8 or w < 8:
            continue

        result.append((key, coeff))

    return result


def phase_difference(current, previous):
    """
    Robust wrapped phase difference.
    """
    z = current * np.conj(previous)

    return np.angle(z)


def resize_to_rows(arr, target_h):
    """
    Resize a pyramid phase map to the original ROI height.
    """
    arr = np.asarray(arr, dtype=np.float32)

    target_w = arr.shape[1]

    return cv2.resize(
        arr,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# LOAD VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video:\n{VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print()
print("============================================================")
print("HYBRID VMSLIM SPEECH RECONSTRUCTION")
print("============================================================")
print(f"Video       : {VIDEO}")
print(f"Resolution  : {width} x {height}")
print(f"FPS         : {fps:.6f}")
print(f"Frames      : {frame_count}")
print(f"RS delay    : {ROW_DELAY:.9e} s/row")
print(f"ROI         : x={X1}:{X2}, y={Y1}:{Y2}")
print("============================================================")
print()

# ============================================================
# READ FIRST FRAME
# ============================================================

ok, first_frame = cap.read()

if not ok:
    raise RuntimeError("Could not read first frame.")

reference_frame = first_frame.copy()

# ROI dimensions
roi_w = X2 - X1
roi_h = Y2 - Y1

# ============================================================
# FIRST PYRAMID
# ============================================================

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

first_roi = first_gray[Y1:Y2, X1:X2]

first_roi = first_roi.astype(np.float32) / 255.0

print("Building reference pyramid...")

reference_pyr = pt.pyramids.SteerablePyramidFreq(
    first_roi,
    height=PYRAMID_HEIGHT,
    order=3,
    is_complex=True
)

reference_bands = get_complex_bands(reference_pyr)

print(f"Complex pyramid bands: {len(reference_bands)}")

if len(reference_bands) == 0:
    raise RuntimeError("No complex pyramid bands found.")

print()

# ============================================================
# STORAGE
# ============================================================

# Each band gets its own row-phase matrix.
#
# matrix shape:
#     frames x rows
#
# We keep bands separate instead of averaging them.

band_names = []
band_data = []

for key, coeff in reference_bands:

    band_names.append(str(key))

    band_data.append({
        "key": key,
        "previous": coeff.copy(),
        "rows": []
    })

print("Tracking individual phase bands...")
print()

# ============================================================
# PROCESS VIDEO
# ============================================================

previous_frame = first_frame

frame_index = 1

while True:

    ok, frame = cap.read()

    if not ok:
        break

    # --------------------------------------------------------
    # 1. GLOBAL MOTION COMPENSATION
    # --------------------------------------------------------

    stabilized = ecc_stabilize(
        reference_frame,
        frame
    )

    # --------------------------------------------------------
    # 2. GRAYSCALE ROI
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        stabilized,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[Y1:Y2, X1:X2]

    roi = roi.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # 3. CURRENT COMPLEX STEERABLE PYRAMID
    # --------------------------------------------------------

    pyr = pt.pyramids.SteerablePyramidFreq(
        roi,
        height=PYRAMID_HEIGHT,
        order=3,
        is_complex=True
    )

    current_bands = dict(
        get_complex_bands(pyr)
    )

    # --------------------------------------------------------
    # 4. PHASE DIFFERENCE FOR EACH BAND
    # --------------------------------------------------------

    for band in band_data:

        key = band["key"]

        if key not in current_bands:
            continue

        current = current_bands[key]
        previous = band["previous"]

        # Phase change
        phase = phase_difference(
            current,
            previous
        )

        # Amplitude weighting
        amplitude = (
            np.abs(current) +
            np.abs(previous)
        ) * 0.5

        weight = amplitude ** 2

        # Avoid excessive amplification of weak coefficients
        weight /= (
            np.mean(weight) + 1e-8
        )

        weight = np.clip(
            weight,
            0,
            20
        )

        weighted_phase = phase * weight

        # Convert pyramid map to original ROI row scale
        resized = cv2.resize(
            weighted_phase.astype(np.float32),
            (roi_w, roi_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Average horizontally.
        #
        # IMPORTANT:
        # We retain the ROW dimension.
        row_signal = np.mean(
            resized,
            axis=1
        )

        band["rows"].append(row_signal)

        band["previous"] = current.copy()

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_index % 25 == 0:

        print(
            f"Processed {frame_index:4d} / "
            f"{frame_count} frames"
        )

    frame_index += 1

cap.release()

print()
print("Video processing complete.")
print()

# ============================================================
# CONVERT EACH BAND TO ROLLING-SHUTTER TIME SERIES
# ============================================================

print("Constructing rolling-shutter time axes...")
print()

candidate_signals = []

for band_index, band in enumerate(band_data):

    if len(band["rows"]) < 10:
        continue

    matrix = np.asarray(
        band["rows"],
        dtype=np.float64
    )

    # matrix:
    #     frame x row

    n_frames, n_rows = matrix.shape

    print(
        f"Band {band_index:02d} "
        f"{band['key']} -> "
        f"{n_frames} x {n_rows}"
    )

    # --------------------------------------------------------
    # Remove row DC
    # --------------------------------------------------------

    matrix -= np.mean(
        matrix,
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # Remove frame-wise mean
    # --------------------------------------------------------

    matrix -= np.mean(
        matrix,
        axis=0,
        keepdims=True
    )

    # --------------------------------------------------------
    # Flatten according to rolling shutter.
    #
    # For frame n and row r:
    #
    # t = n/fps + r * ROW_DELAY
    # --------------------------------------------------------

    values = matrix.ravel()

    frame_numbers = np.repeat(
        np.arange(n_frames),
        n_rows
    )

    row_numbers = np.tile(
        np.arange(n_rows),
        n_frames
    )

    times = (
        frame_numbers / fps +
        row_numbers * ROW_DELAY
    )

    # --------------------------------------------------------
    # Remove duplicate / non-increasing timestamps
    # --------------------------------------------------------

    order = np.argsort(times)

    times = times[order]
    values = values[order]

    keep = np.concatenate(
        ([True], np.diff(times) > 0)
    )

    times = times[keep]
    values = values[keep]

    # --------------------------------------------------------
    # Interpolate to regular 48 kHz audio grid
    # --------------------------------------------------------

    t_audio = np.arange(
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

    audio = interpolator(t_audio)

    # --------------------------------------------------------
    # Speech-band filter
    # --------------------------------------------------------

    audio = bandpass(
        audio,
        AUDIO_FS,
        LOW_HZ,
        HIGH_HZ
    )

    audio -= np.mean(audio)

    # --------------------------------------------------------
    # Normalize temporarily for scoring
    # --------------------------------------------------------

    audio_norm = normalize_audio(audio)

    candidate_signals.append({
        "index": band_index,
        "key": band["key"],
        "audio": audio_norm
    })


# ============================================================
# CLASSICAL CANDIDATE SCORING
# ============================================================

print()
print("============================================================")
print("SCORING INDIVIDUAL PHASE CANDIDATES")
print("============================================================")

def candidate_score(x, fs):

    # --------------------------------------------------------
    # Speech-frequency energy
    # --------------------------------------------------------

    f, pxx = signal.welch(
        x,
        fs=fs,
        nperseg=min(
            16384,
            len(x)
        )
    )

    speech_band = (
        (f >= 250) &
        (f <= 4000)
    )

    low_band = (
        (f >= 70) &
        (f < 250)
    )

    speech_energy = np.sum(
        pxx[speech_band]
    )

    low_energy = np.sum(
        pxx[low_band]
    ) + 1e-12

    spectral_ratio = (
        speech_energy /
        low_energy
    )

    # --------------------------------------------------------
    # Modulation energy in approximately speech-rate region
    # --------------------------------------------------------

    env = np.abs(
        signal.hilbert(x)
    )

    env = env - np.mean(env)

    fm, pm = signal.welch(
        env,
        fs=fs,
        nperseg=min(
            32768,
            len(env)
        )
    )

    modulation = (
        (fm >= 2) &
        (fm <= 20)
    )

    modulation_energy = np.sum(
        pm[modulation]
    )

    # --------------------------------------------------------
    # Avoid candidates dominated by one narrow tone
    # --------------------------------------------------------

    total_energy = np.sum(pxx) + 1e-12

    peak_ratio = (
        np.max(pxx[speech_band]) /
        total_energy
    )

    # Larger is better for speech-band/modulation energy.
    # Smaller is better for a single dominant spectral spike.
    score = (
        np.log10(spectral_ratio + 1e-12)
        +
        0.5 * np.log10(
            modulation_energy + 1e-12
        )
        -
        2.0 * peak_ratio
    )

    return score


scored = []

for candidate in candidate_signals:

    score = candidate_score(
        candidate["audio"],
        AUDIO_FS
    )

    candidate["score"] = score

    scored.append(candidate)

scored.sort(
    key=lambda z: z["score"],
    reverse=True
)

print()

for rank, candidate in enumerate(
    scored[:10],
    start=1
):

    print(
        f"{rank:02d}. "
        f"Band {candidate['index']:02d} "
        f"{candidate['key']} "
        f"score={candidate['score']:.4f}"
    )

# ============================================================
# SAVE TOP INDIVIDUAL CANDIDATES
# ============================================================

print()
print("Saving candidate WAV files...")

TOP_N = min(
    8,
    len(scored)
)

for rank, candidate in enumerate(
    scored[:TOP_N],
    start=1
):

    path = (
        OUT_DIR /
        f"candidate_{rank:02d}_"
        f"band_{candidate['index']:02d}.wav"
    )

    save_wav(
        path,
        candidate["audio"]
    )

    print(
        f"Saved: {path.name}"
    )

# ============================================================
# BUILD COMBINED SIGNAL FROM TOP CANDIDATES
# ============================================================

print()
print("Building combined reconstruction...")

if len(scored) == 0:
    raise RuntimeError(
        "No candidate signals were produced."
    )

# Use top candidates.
#
# Instead of simply averaging everything,
# weight the stronger candidates more heavily.

top = scored[:TOP_N]

length = min(
    len(c["audio"])
    for c in top
)

combined = np.zeros(
    length,
    dtype=np.float64
)

weights = []

for candidate in top:

    # Convert score into positive weight.
    weights.append(
        np.exp(
            candidate["score"] -
            top[-1]["score"]
        )
    )

weights = np.asarray(weights)

weights /= (
    np.sum(weights) + 1e-12
)

for weight, candidate in zip(
    weights,
    top
):

    combined += (
        weight *
        candidate["audio"][:length]
    )

# ------------------------------------------------------------
# Final classical speech filtering
# ------------------------------------------------------------

combined = bandpass(
    combined,
    AUDIO_FS,
    80,
    5000
)

# Gentle dynamic compression
combined = np.tanh(
    1.5 * combined
)

combined = normalize_audio(
    combined
)

combined_path = (
    OUT_DIR /
    "A15_speech_hybrid_combined.wav"
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

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.specgram(
    combined,
    NFFT=2048,
    Fs=AUDIO_FS,
    noverlap=1536
)

plt.ylim(0, 6000)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title(
    "Hybrid Visual Microphone Speech Reconstruction"
)

plt.colorbar(
    label="Power"
)

spectrogram_path = (
    OUT_DIR /
    "A15_speech_hybrid_spectrogram.png"
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
print("============================================================")
print("DONE")
print("============================================================")
print()
print(f"Output folder:")
print(OUT_DIR)
print()
print("LISTEN TO THESE FIRST:")
print("  candidate_01_*.wav")
print("  candidate_02_*.wav")
print("  candidate_03_*.wav")
print("  candidate_04_*.wav")
print("  A15_speech_hybrid_combined.wav")
print()