import cv2
import numpy as np
import pyrtools as pt
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_test_clean.mp4"

# Refined 10 Hz rolling-shutter calibration
ROW_DELAY = 7.7435e-6

EXPECTED_FREQ = 440.0

# Central region containing the bedsheet
X1_FRAC = 0.15
X2_FRAC = 0.85

Y1_FRAC = 0.10
Y2_FRAC = 0.90

# Steerable pyramid
HEIGHT = 4
ORDER = 3

# Temporal frequency search
F_MIN = 300.0
F_MAX = 600.0
F_STEP = 0.25


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open:\n{VIDEO}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n==========================================")
print("VMSLIM-STYLE STEERABLE PHASE TEST")
print("==========================================")

print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.6f}")
print(f"Frames     : {nframes}")
print(f"Duration   : {nframes / fps:.4f} s")
print(f"Row delay  : {ROW_DELAY * 1e6:.4f} us")
print(f"Expected   : {EXPECTED_FREQ:.2f} Hz")


# ============================================================
# ROI
# ============================================================

x1 = int(width * X1_FRAC)
x2 = int(width * X2_FRAC)

y1 = int(height * Y1_FRAC)
y2 = int(height * Y2_FRAC)

print(
    f"ROI        : "
    f"x={x1}:{x2}, y={y1}:{y2}"
)


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

roi = gray[y1:y2, x1:x2]

# Normalize
roi = roi / 255.0

# Build complex steerable pyramid
pyr_ref = pt.pyramids.SteerablePyramidFreq(
    roi,
    height=HEIGHT,
    order=ORDER,
    is_complex=True
)

print(
    f"Pyramid bands : "
    f"{len(pyr_ref.pyr_coeffs)}"
)


# ============================================================
# REFERENCE PHASE
# ============================================================

reference_phase = {}
reference_amp = {}

for key, coeff in pyr_ref.pyr_coeffs.items():

    if key == "residual_highpass":
        continue

    if key == "residual_lowpass":
        continue

    reference_phase[key] = np.angle(coeff)

    reference_amp[key] = np.abs(coeff)


# ============================================================
# DETERMINE SPATIAL BANDS
# ============================================================

band_keys = list(
    reference_phase.keys()
)

print("\nUsing pyramid bands:")

for key in band_keys:
    print(" ", key)


# ============================================================
# READ ALL FRAMES
# ============================================================

phase_band_signals = {
    key: []
    for key in band_keys
}


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
            gray[y1:y2, x1:x2]
            / 255.0
        )


    # --------------------------------------------------------
    # Build complex steerable pyramid
    # --------------------------------------------------------

    pyr = pt.pyramids.SteerablePyramidFreq(
        current,
        height=HEIGHT,
        order=ORDER,
        is_complex=True
    )


    # --------------------------------------------------------
    # Extract phase difference
    # --------------------------------------------------------

    for key in band_keys:

        coeff = pyr.pyr_coeffs[key]

        ref_coeff_phase = (
            reference_phase[key]
        )

        current_phase = np.angle(
            coeff
        )

        # Wrapped phase difference
        delta = np.angle(
            np.exp(
                1j *
                (
                    current_phase -
                    ref_coeff_phase
                )
            )
        )


        # ----------------------------------------------------
        # Amplitude weighting
        # ----------------------------------------------------

        amplitude = np.abs(coeff)

        weight = amplitude ** 2

        # Weighted circular mean
        z = (
            np.sum(
                weight *
                np.exp(1j * delta)
            )
            /
            (
                np.sum(weight) +
                1e-12
            )
        )

        phase_value = np.angle(z)

        phase_band_signals[key].append(
            phase_value
        )


    if frame_idx % 25 == 0:

        print(
            f"Processed "
            f"{frame_idx}/{nframes}"
        )


cap.release()


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

for key in band_keys:

    phase_band_signals[key] = np.asarray(
        phase_band_signals[key]
    )


# ============================================================
# DISPLAY BAND INFORMATION
# ============================================================

print("\nPhase-band signals:")

for key in band_keys:

    sig = phase_band_signals[key]

    print(
        f"{key}: "
        f"{len(sig)} samples, "
        f"std={np.std(sig):.6e}"
    )


# ============================================================
# FRAME-DOMAIN PHASE ANALYSIS
# ============================================================

# First determine which pyramid bands contain
# meaningful temporal variation.

frame_time = (
    np.arange(nframes) / fps
)


def frame_frequency_score(
    signal,
    freq
):

    omega = 2 * np.pi * freq

    A = np.column_stack([
        np.cos(omega * frame_time),
        np.sin(omega * frame_time),
        np.ones_like(frame_time)
    ])

    coeff, _, _, _ = np.linalg.lstsq(
        A,
        signal,
        rcond=None
    )

    fitted = A @ coeff

    return np.sqrt(
        coeff[0] ** 2 +
        coeff[1] ** 2
    )


# ============================================================
# COMBINE PYRAMID BANDS
# ============================================================

band_matrix = []

for key in band_keys:

    sig = phase_band_signals[key]

    sig = sig - np.mean(sig)

    std = np.std(sig)

    if std > 1e-12:

        sig = sig / std

        band_matrix.append(sig)


if len(band_matrix) == 0:

    raise RuntimeError(
        "No usable phase bands found."
    )


band_matrix = np.asarray(
    band_matrix
)

combined_phase = np.mean(
    band_matrix,
    axis=0
)


print(
    f"\nCombined phase signal: "
    f"{combined_phase.shape}"
)


# ============================================================
# IMPORTANT:
# FRAME-DOMAIN 440 Hz IS NOT VALID.
#
# We use this only to identify the strongest
# spatial pyramid bands, not to claim 440 Hz recovery.
# ============================================================


# ============================================================
# BUILD ROW-WISE PHASE SIGNAL
# ============================================================

# The steerable pyramid coefficients are spatially
# downsampled at different scales.
#
# For this first VMSlim validation, we construct a
# spatial phase map from all usable bands and resize
# each band back to the ROI resolution.


row_phase_frames = []


print("\nReconstructing spatial phase maps...")


# Re-open video
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        "Could not reopen video."
    )


# Skip reference frame
ret, frame = cap.read()


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
            gray[y1:y2, x1:x2]
            / 255.0
        )


    pyr = pt.pyramids.SteerablePyramidFreq(
        current,
        height=HEIGHT,
        order=ORDER,
        is_complex=True
    )


    weighted_maps = []
    weights_maps = []


    for key in band_keys:

        coeff = pyr.pyr_coeffs[key]

        ref_phase = reference_phase[key]

        phase = np.angle(coeff)

        delta = np.angle(
            np.exp(
                1j *
                (
                    phase -
                    ref_phase
                )
            )
        )

        amplitude = np.abs(coeff)

        weight = amplitude ** 2

        # Resize phase and weight to ROI
        h_roi, w_roi = current.shape

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
            weight_up *
            np.sin(delta_up)
        )

        weights_maps.append(
            weight_up
        )


    numerator = np.sum(
        weighted_maps,
        axis=0
    )

    denominator = (
        np.sum(
            weights_maps,
            axis=0
        ) +
        1e-12
    )

    phase_map = (
        numerator /
        denominator
    )


    # Average horizontally to obtain
    # one signal value per image row.

    row_signal = np.mean(
        phase_map,
        axis=1
    )

    row_phase_frames.append(
        row_signal
    )


    if frame_idx % 25 == 0:

        print(
            f"Spatial phase "
            f"{frame_idx}/{nframes}"
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
# REMOVE ROW DC
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

ROW_BIN = 4

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


# ============================================================
# ROLLING-SHUTTER TIMESTAMPS
# ============================================================

frame_times = (
    np.arange(
        nframes
    ) / fps
)[:, None]

row_offsets = (
    row_positions *
    ROW_DELAY
)[None, :]


times = (
    frame_times +
    row_offsets
)

signals = (
    row_values.T
)


# Flatten
times = times.ravel()
signals = signals.ravel()


# Sort
order = np.argsort(times)

times = times[order]
signals = signals[order]


# Normalize
signals -= np.mean(signals)

std = np.std(signals)

if std > 0:
    signals /= std


print(
    f"\nFinal samples : "
    f"{len(signals)}"
)


# ============================================================
# FREQUENCY SEARCH
# ============================================================

def fit_frequency(freq):

    omega = 2 * np.pi * freq

    A = np.column_stack([
        np.cos(omega * times),
        np.sin(omega * times),
        np.ones_like(times)
    ])

    coeff, _, _, _ = np.linalg.lstsq(
        A,
        signals,
        rcond=None
    )

    fitted = A @ coeff

    residual = (
        signals -
        fitted
    )

    signal_power = np.var(
        fitted
    )

    noise_power = np.var(
        residual
    )

    if noise_power <= 1e-20:

        snr = np.inf

    else:

        snr = 10 * np.log10(
            signal_power /
            noise_power
        )

    amplitude = np.sqrt(
        coeff[0] ** 2 +
        coeff[1] ** 2
    )

    return amplitude, snr


frequencies = np.arange(
    300,
    601,
    0.25
)

amplitudes = []
snrs = []


print(
    "\nSearching phase-reconstructed signal..."
)


for f in frequencies:

    amp, snr = fit_frequency(f)

    amplitudes.append(amp)
    snrs.append(snr)


amplitudes = np.asarray(
    amplitudes
)

snrs = np.asarray(
    snrs
)


# ============================================================
# RESULT
# ============================================================

best = np.argmax(
    amplitudes
)

detected = frequencies[best]

amp440, snr440 = fit_frequency(
    EXPECTED_FREQ
)


print("\n==========================================")
print("VMSLIM PHASE RESULT")
print("==========================================")

print(
    f"Expected frequency : "
    f"{EXPECTED_FREQ:.2f} Hz"
)

print(
    f"Detected frequency : "
    f"{detected:.2f} Hz"
)

print(
    f"Frequency error     : "
    f"{abs(detected - EXPECTED_FREQ):.2f} Hz"
)

print(
    f"Amplitude at 440 Hz : "
    f"{amp440:.6e}"
)

print(
    f"SNR at 440 Hz       : "
    f"{snr440:.2f} dB"
)


# ============================================================
# TOP RESULTS
# ============================================================

top = np.argsort(
    amplitudes
)[::-1][:10]

print(
    "\nTop VMSlim phase frequencies:"
)

for idx in top:

    print(
        f"{frequencies[idx]:8.2f} Hz   "
        f"Amplitude={amplitudes[idx]:.6e}   "
        f"SNR={snrs[idx]:7.2f} dB"
    )


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    frequencies,
    amplitudes
)

plt.axvline(
    EXPECTED_FREQ,
    linestyle="--",
    label="Expected 440 Hz"
)

plt.axvline(
    detected,
    linestyle=":",
    label=f"Detected {detected:.2f} Hz"
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Phase amplitude"
)

plt.title(
    "VMSlim-Style Complex Steerable Pyramid Phase"
)

plt.grid(True)

plt.legend()

plt.tight_layout()


output_plot = (
    r"C:\Users\Harkiran\Downloads\Mobile Devices"
    r"\A15_440Hz_vmslim_phase.png"
)

plt.savefig(
    output_plot,
    dpi=200
)

plt.show()


# ============================================================
# SAVE
# ============================================================

output_data = (
    r"C:\Users\Harkiran\Downloads\Mobile Devices"
    r"\A15_440Hz_vmslim_phase.npz"
)

np.savez(
    output_data,
    time=times,
    signal=signals,
    frequencies=frequencies,
    amplitudes=amplitudes,
    snrs=snrs,
    row_delay=ROW_DELAY,
    fps=fps
)

print("\nSaved:")
print(output_plot)
print(output_data)