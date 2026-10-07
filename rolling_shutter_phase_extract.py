import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt


# ============================================================
# PHASE-BASED ROLLING-SHUTTER AUDIO EXTRACTION
# Samsung Galaxy A15
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_test_clean.mp4"

# From the controlled rolling-shutter calibration
ROW_DELAY = 8.3763e-6       # seconds / row

TARGET_FREQ = 440.0         # known test tone

OUTPUT_FS = 48000

# Number of horizontal row bands
NUM_BANDS = 240

# Spatial region used for analysis
TOP_MARGIN = 0.10
BOTTOM_MARGIN = 0.90

LEFT_MARGIN = 0.15
RIGHT_MARGIN = 0.85

# Process every frame
MAX_FRAMES = None


# ============================================================
# RIESZ / MONOGENIC PHASE
# ============================================================

def monogenic_phase(image):
    """
    Estimate local phase using a 2-D Riesz transform.

    The returned phase is useful for measuring small
    image-motion / vibration changes.
    """

    image = image.astype(np.float64)

    # Remove image DC component
    image = image - np.mean(image)

    h, w = image.shape

    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)

    FX, FY = np.meshgrid(fx, fy)

    magnitude = np.sqrt(FX ** 2 + FY ** 2)

    magnitude[0, 0] = 1.0

    F = np.fft.fft2(image)

    # Riesz filters
    Rx_filter = -1j * FX / magnitude
    Ry_filter = -1j * FY / magnitude

    Rx_filter[0, 0] = 0
    Ry_filter[0, 0] = 0

    rx = np.real(
        np.fft.ifft2(F * Rx_filter)
    )

    ry = np.real(
        np.fft.ifft2(F * Ry_filter)
    )

    # Local amplitude
    amplitude = np.sqrt(
        image ** 2 +
        rx ** 2 +
        ry ** 2
    )

    # Monogenic phase
    phase = np.arctan2(
        np.sqrt(rx ** 2 + ry ** 2),
        image
    )

    return phase, amplitude


# ============================================================
# PHASE WRAPPING
# ============================================================

def wrapped_phase_difference(a, b):
    """
    Calculate phase difference b-a while keeping
    the result inside [-pi, pi].
    """

    return np.angle(
        np.exp(1j * (b - a))
    )


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = frame_count / fps

print()
print("=" * 60)
print("PHASE-BASED ROLLING-SHUTTER EXTRACTION")
print("=" * 60)

print(f"Video       : {VIDEO}")
print(f"Resolution  : {width} x {height}")
print(f"FPS         : {fps:.6f}")
print(f"Frames      : {frame_count}")
print(f"Duration    : {duration:.4f} s")
print(f"Row delay   : {ROW_DELAY * 1e6:.4f} us")
print(f"Target tone : {TARGET_FREQ:.2f} Hz")


# ============================================================
# ROI
# ============================================================

y_start = int(height * TOP_MARGIN)
y_end = int(height * BOTTOM_MARGIN)

x_start = int(width * LEFT_MARGIN)
x_end = int(width * RIGHT_MARGIN)

roi_height = y_end - y_start
roi_width = x_end - x_start

print()
print("ROI")
print(f"X: {x_start} - {x_end}")
print(f"Y: {y_start} - {y_end}")
print(f"Size: {roi_width} x {roi_height}")


# ============================================================
# ROW BANDS
# ============================================================

row_edges = np.linspace(
    y_start,
    y_end,
    NUM_BANDS + 1
).astype(int)

row_centers = (
    row_edges[:-1] +
    row_edges[1:]
) // 2

print(f"Row bands   : {NUM_BANDS}")


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, frame = cap.read()

if not ret:
    raise RuntimeError(
        "Could not read first frame."
    )

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
).astype(np.float64)

roi = gray[
    y_start:y_end,
    x_start:x_end
]

print()
print("Calculating reference phase...")

reference_phase, reference_amp = monogenic_phase(
    roi
)


# ============================================================
# ROW-WISE REFERENCE AMPLITUDE
# ============================================================

row_weights = np.zeros(
    NUM_BANDS,
    dtype=np.float64
)

for i in range(NUM_BANDS):

    a = row_edges[i] - y_start
    b = row_edges[i + 1] - y_start

    row_weights[i] = np.mean(
        reference_amp[a:b]
    )


# Normalize weights
weight_scale = np.median(
    row_weights[row_weights > 0]
)

if weight_scale > 0:
    row_weights = row_weights / weight_scale


# ============================================================
# PROCESS FRAMES
# ============================================================

row_phase_signals = []

frame_index = 0

# Include first frame
first_signal = np.zeros(
    NUM_BANDS,
    dtype=np.float64
)

row_phase_signals.append(
    first_signal
)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_index += 1

    if (
        MAX_FRAMES is not None
        and frame_index >= MAX_FRAMES
    ):
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float64)

    roi = gray[
        y_start:y_end,
        x_start:x_end
    ]

    current_phase, current_amp = monogenic_phase(
        roi
    )

    # Phase relative to first frame
    delta_phase = wrapped_phase_difference(
        reference_phase,
        current_phase
    )

    row_signal = np.zeros(
        NUM_BANDS,
        dtype=np.float64
    )

    for i in range(NUM_BANDS):

        a = row_edges[i] - y_start
        b = row_edges[i + 1] - y_start

        phase_band = delta_phase[a:b]
        amp_band = current_amp[a:b]

        # Amplitude weighting suppresses
        # nearly textureless regions.
        weight = amp_band

        numerator = np.sum(
            phase_band * weight
        )

        denominator = np.sum(weight)

        if denominator > 1e-12:

            row_signal[i] = (
                numerator /
                denominator
            )

        else:

            row_signal[i] = 0.0

    row_phase_signals.append(
        row_signal
    )

    if frame_index % 25 == 0:

        print(
            f"Processed "
            f"{frame_index + 1}/{frame_count} frames"
        )


cap.release()

row_phase_signals = np.asarray(
    row_phase_signals
)

print()
print("Phase matrix:")
print(
    "Frames x RowBands =",
    row_phase_signals.shape
)


# ============================================================
# REMOVE ROW-WISE DC
# ============================================================

row_phase_signals -= np.mean(
    row_phase_signals,
    axis=0,
    keepdims=True
)


# ============================================================
# BUILD ROLLING-SHUTTER TIME AXIS
# ============================================================

timestamps = []
values = []

nframes = row_phase_signals.shape[0]

for f in range(nframes):

    frame_start = f / fps

    for r in range(NUM_BANDS):

        absolute_row = row_centers[r]

        # Time at which this row was exposed
        row_time = (
            frame_start +
            absolute_row * ROW_DELAY
        )

        timestamps.append(row_time)

        values.append(
            row_phase_signals[f, r]
        )


timestamps = np.asarray(
    timestamps
)

values = np.asarray(
    values
)


# ============================================================
# SORT BY TIME
# ============================================================

order = np.argsort(timestamps)

timestamps = timestamps[order]
values = values[order]


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

valid = (
    np.isfinite(timestamps)
    &
    np.isfinite(values)
)

timestamps = timestamps[valid]
values = values[valid]


# ============================================================
# REMOVE MEAN
# ============================================================

values -= np.mean(values)


# ============================================================
# NORMALIZE
# ============================================================

std = np.std(values)

if std > 0:

    values /= std


# ============================================================
# UNIFORM HIGH-RATE GRID
# ============================================================

t0 = timestamps[0]
t1 = timestamps[-1]

uniform_t = np.arange(
    t0,
    t1,
    1.0 / OUTPUT_FS
)

print()
print("Reconstructing high-rate signal...")
print(
    f"Output sampling rate : "
    f"{OUTPUT_FS} Hz"
)

audio = np.interp(
    uniform_t,
    timestamps,
    values
)


# ============================================================
# REMOVE DC
# ============================================================

audio -= np.mean(audio)


# ============================================================
# NORMALIZE
# ============================================================

peak = np.max(
    np.abs(audio)
)

if peak > 0:

    audio /= peak


# ============================================================
# 300–600 Hz DIAGNOSTIC FILTER
# ============================================================

sos = butter(
    4,
    [300.0, 600.0],
    btype="bandpass",
    fs=OUTPUT_FS,
    output="sos"
)

audio_440 = sosfiltfilt(
    sos,
    audio
)

peak = np.max(
    np.abs(audio_440)
)

if peak > 0:

    audio_440 /= peak


# ============================================================
# SAVE WAV
# ============================================================

output_dir = os.path.dirname(
    VIDEO
)

full_wav = os.path.join(
    output_dir,
    "A15_phase_rolling_recovered_full.wav"
)

tone_wav = os.path.join(
    output_dir,
    "A15_phase_rolling_recovered_300_600Hz.wav"
)

wavfile.write(
    full_wav,
    OUTPUT_FS,
    np.int16(
        np.clip(
            audio,
            -1,
            1
        ) * 32767
    )
)

wavfile.write(
    tone_wav,
    OUTPUT_FS,
    np.int16(
        np.clip(
            audio_440,
            -1,
            1
        ) * 32767
    )
)


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

print()
print("=" * 60)
print("SPECTRAL ANALYSIS")
print("=" * 60)

nperseg = min(
    262144,
    len(audio)
)

freqs, power = welch(
    audio,
    fs=OUTPUT_FS,
    nperseg=nperseg
)


# ============================================================
# FIND PEAK NEAR 440 Hz
# ============================================================

search = (
    (freqs >= 300)
    &
    (freqs <= 600)
)

local_freqs = freqs[search]
local_power = power[search]

peak_index = np.argmax(
    local_power
)

detected_freq = (
    local_freqs[peak_index]
)

detected_power = (
    local_power[peak_index]
)


# ============================================================
# POWER AT EXACT 440 Hz
# ============================================================

target_index = np.argmin(
    np.abs(
        freqs - TARGET_FREQ
    )
)

target_power = power[
    target_index
]


# ============================================================
# LOCAL NOISE FLOOR
# ============================================================

noise_region = (
    (freqs >= 300)
    &
    (freqs <= 600)
    &
    (
        np.abs(
            freqs - TARGET_FREQ
        ) > 20
    )
)

noise_floor = np.median(
    power[noise_region]
)

if noise_floor > 0:

    snr_db = 10 * np.log10(
        target_power /
        noise_floor
    )

else:

    snr_db = float("inf")


# ============================================================
# PRINT RESULT
# ============================================================

print()
print(
    f"Expected frequency : "
    f"{TARGET_FREQ:.2f} Hz"
)

print(
    f"Detected peak      : "
    f"{detected_freq:.2f} Hz"
)

print(
    f"Approx. SNR        : "
    f"{snr_db:.2f} dB"
)


frequency_error = abs(
    detected_freq -
    TARGET_FREQ
)

print(
    f"Frequency error     : "
    f"{frequency_error:.2f} Hz"
)


if frequency_error <= 10:

    print()
    print(
        ">>> PEAK NEAR 440 Hz DETECTED <<<"
    )

else:

    print()
    print(
        ">>> NO CLEAR 440 Hz PEAK <<<"
    )


# ============================================================
# SAVE SPECTRUM
# ============================================================

spectrum_csv = os.path.join(
    output_dir,
    "A15_phase_rolling_spectrum.csv"
)

np.savetxt(
    spectrum_csv,
    np.column_stack(
        [freqs, power]
    ),
    delimiter=",",
    header="frequency_hz,power",
    comments=""
)


# ============================================================
# SAVE DIAGNOSTIC PLOT
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    freqs,
    power
)

plt.xlim(
    0,
    1000
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Power"
)

plt.title(
    "Phase-Based Rolling-Shutter Spectrum"
)

plt.axvline(
    TARGET_FREQ,
    linestyle="--",
    label="Expected 440 Hz"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plot_path = os.path.join(
    output_dir,
    "A15_phase_rolling_spectrum.png"
)

plt.tight_layout()

plt.savefig(
    plot_path,
    dpi=150
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("OUTPUT")
print("=" * 60)

print(
    "Full phase reconstruction:"
)

print(full_wav)

print()
print(
    "300–600 Hz phase reconstruction:"
)

print(tone_wav)

print()
print(
    "Spectrum CSV:"
)

print(spectrum_csv)

print()
print(
    "Spectrum plot:"
)

print(plot_path)

print()
print("Done.")