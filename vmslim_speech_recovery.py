import cv2
import numpy as np
import pyrtools as pt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt, welch
from scipy.io.wavfile import write


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\itsybitsy.mp4"

# Refined 10 Hz rolling-shutter calibration
ROW_DELAY = 7.7435e-6

# Central ROI
X1_FRAC = 0.15
X2_FRAC = 0.85
Y1_FRAC = 0.10
Y2_FRAC = 0.90

# Steerable pyramid
HEIGHT = 4
ORDER = 3

# Reconstruction
ROW_BIN = 4
OUTPUT_SR = 48000

# Speech band
LOWCUT = 80.0
HIGHCUT = 8000.0


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open:\n{VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n==========================================")
print("VMSLIM-STYLE SPEECH RECOVERY")
print("==========================================")

print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.6f}")
print(f"Frames     : {nframes}")
print(f"Duration   : {nframes / fps:.4f} s")
print(f"Row delay  : {ROW_DELAY * 1e6:.4f} us")


# ============================================================
# ROI
# ============================================================

x1 = int(width * X1_FRAC)
x2 = int(width * X2_FRAC)

y1 = int(height * Y1_FRAC)
y2 = int(height * Y2_FRAC)

print(f"ROI        : x={x1}:{x2}, y={y1}:{y2}")


# ============================================================
# FIRST FRAME
# ============================================================

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read first frame")

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
).astype(np.float64)

roi = gray[y1:y2, x1:x2] / 255.0


# ============================================================
# REFERENCE PYRAMID
# ============================================================

pyr_ref = pt.pyramids.SteerablePyramidFreq(
    roi,
    height=HEIGHT,
    order=ORDER,
    is_complex=True
)

reference_phase = {}

for key, coeff in pyr_ref.pyr_coeffs.items():

    if key in ["residual_highpass", "residual_lowpass"]:
        continue

    reference_phase[key] = np.angle(coeff)

band_keys = list(reference_phase.keys())

print(f"Pyramid bands : {len(band_keys)}")


# ============================================================
# PROCESS VIDEO
# ============================================================

row_phase_frames = []

print("\nProcessing frames...")

for frame_idx in range(nframes):

    if frame_idx == 0:

        current = roi

    else:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float64)

        current = (
            gray[y1:y2, x1:x2] / 255.0
        )

    pyr = pt.pyramids.SteerablePyramidFreq(
        current,
        height=HEIGHT,
        order=ORDER,
        is_complex=True
    )

    weighted_maps = []
    weights_maps = []

    h_roi, w_roi = current.shape

    for key in band_keys:

        coeff = pyr.pyr_coeffs[key]

        phase = np.angle(coeff)

        ref_phase = reference_phase[key]

        delta = np.angle(
            np.exp(
                1j * (phase - ref_phase)
            )
        )

        amplitude = np.abs(coeff)

        weight = amplitude ** 2

        delta_up = cv2.resize(
            delta.astype(np.float32),
            (w_roi, h_roi),
            interpolation=cv2.INTER_LINEAR
        )

        weight_up = cv2.resize(
            weight.astype(np.float32),
            (w_roi, h_roi),
            interpolation=cv2.INTER_LINEAR
        )

        weighted_maps.append(
            weight_up * np.sin(delta_up)
        )

        weights_maps.append(weight_up)

    numerator = np.sum(
        weighted_maps,
        axis=0
    )

    denominator = (
        np.sum(weights_maps, axis=0)
        + 1e-12
    )

    phase_map = numerator / denominator

    # One phase value per image row
    row_signal = np.mean(
        phase_map,
        axis=1
    )

    row_phase_frames.append(row_signal)

    if frame_idx % 25 == 0:
        print(
            f"Processed {frame_idx}/{nframes}"
        )

cap.release()

row_phase_frames = np.asarray(
    row_phase_frames
)

print(
    "\nRow phase matrix:",
    row_phase_frames.shape
)


# ============================================================
# REMOVE TEMPORAL DC
# ============================================================

row_phase_frames -= np.mean(
    row_phase_frames,
    axis=0,
    keepdims=True
)


# ============================================================
# ROW VALIDITY
# ============================================================

row_std = np.std(
    row_phase_frames,
    axis=0
)

threshold = np.percentile(
    row_std,
    30
)

valid = row_std > threshold


# ============================================================
# ROW BINNING
# ============================================================

row_positions = []
row_values = []

for start in range(
    0,
    row_phase_frames.shape[1] - ROW_BIN + 1,
    ROW_BIN
):

    end = start + ROW_BIN

    indices = np.arange(
        start,
        end
    )

    if not np.any(valid[indices]):
        continue

    row_positions.append(
        np.mean(indices)
    )

    row_values.append(
        np.mean(
            row_phase_frames[:, indices],
            axis=1
        )
    )

row_positions = np.asarray(
    row_positions
)

row_values = np.asarray(
    row_values
)

print(
    f"Valid row bins : {len(row_positions)}"
)


# ============================================================
# ROLLING-SHUTTER TIMESTAMPS
# ============================================================

frame_times = (
    np.arange(
        row_phase_frames.shape[0]
    ) / fps
)[:, None]

row_offsets = (
    row_positions * ROW_DELAY
)[None, :]

times = (
    frame_times +
    row_offsets
)

signals = row_values.T


# ============================================================
# FLATTEN + SORT
# ============================================================

times = times.ravel()
signals = signals.ravel()

order = np.argsort(times)

times = times[order]
signals = signals[order]


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

mask = np.isfinite(times) & np.isfinite(signals)

times = times[mask]
signals = signals[mask]


# ============================================================
# REMOVE MEAN
# ============================================================

signals -= np.mean(signals)

std = np.std(signals)

if std > 0:
    signals /= std


print(
    f"Non-uniform samples : {len(signals)}"
)

print(
    f"Time range : "
    f"{times[0]:.6f} - {times[-1]:.6f} s"
)


# ============================================================
# INTERPOLATE TO AUDIO SAMPLE RATE
# ============================================================

audio_time = np.arange(
    times[0],
    times[-1],
    1.0 / OUTPUT_SR
)

interpolator = interp1d(
    times,
    signals,
    kind="linear",
    bounds_error=False,
    fill_value=0.0
)

audio = interpolator(
    audio_time
)

audio -= np.mean(audio)


# ============================================================
# PHASE/VIBRATION GAIN EXPERIMENT
# ============================================================

GAINS = [1, 10, 25, 50]

nyquist = OUTPUT_SR / 2.0

sos = butter(
    6,
    [
        LOWCUT / nyquist,
        HIGHCUT / nyquist
    ],
    btype="bandpass",
    output="sos"
)

print("\n==========================================")
print("PHASE GAIN EXPERIMENT")
print("==========================================")

for gain in GAINS:

    print(f"\nProcessing gain = {gain}x")

    # Amplify the reconstructed visual vibration
    gained = audio * gain

    gained -= np.mean(gained)

    # Speech-band filtering
    audio_filtered = sosfiltfilt(
        sos,
        gained
    )

    # Normalize only for WAV output
    peak = np.max(
        np.abs(audio_filtered)
    )

    if peak > 0:
        audio_normalized = (
            0.95 *
            audio_filtered /
            peak
        )
    else:
        audio_normalized = audio_filtered

    # --------------------------------------------------------
    # WAV
    # --------------------------------------------------------

    wav_path = (
        r"C:\Users\Harkiran\Downloads\Mobile Devices"
        rf"\A15_speech_gain_{gain}x.wav"
    )

    write(
        wav_path,
        OUTPUT_SR,
        np.int16(
            audio_normalized * 32767
        )
    )

    # --------------------------------------------------------
    # SPECTROGRAM
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.specgram(
        audio_normalized,
        NFFT=2048,
        Fs=OUTPUT_SR,
        noverlap=1536
    )

    plt.ylim(
        0,
        8000
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.title(
        f"Recovered Speech - "
        f"Phase/Vibration Gain {gain}x"
    )

    plt.colorbar(
        label="Intensity"
    )

    plt.tight_layout()

    spec_path = (
        r"C:\Users\Harkiran\Downloads\Mobile Devices"
        rf"\A15_speech_gain_{gain}x_spectrogram.png"
    )

    plt.savefig(
        spec_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    freqs, psd = welch(
        audio_normalized,
        fs=OUTPUT_SR,
        nperseg=8192
    )

    mask = (
        (freqs >= LOWCUT) &
        (freqs <= HIGHCUT)
    )

    if np.any(mask):

        idx = np.argmax(
            psd[mask]
        )

        dominant = freqs[mask][idx]

    else:

        dominant = np.nan

    print(
        f"Gain {gain}x:"
    )

    print(
        f"  Dominant frequency = "
        f"{dominant:.2f} Hz"
    )

    print(
        f"  WAV = {wav_path}"
    )

    print(
        f"  Spectrogram = {spec_path}"
    )


print("\n==========================================")
print("GAIN EXPERIMENT COMPLETE")
print("==========================================")