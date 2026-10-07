import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_test_clean.mp4"

# Refined 10 Hz rolling-shutter calibration
ROW_DELAY = 7.7435e-6

EXPECTED_FREQ = 440.0

# Image ROI
X1_FRAC = 0.15
X2_FRAC = 0.85

# Small row bins
ROW_BIN = 4

# Gaussian smoothing for spatial phase extraction
SIGMA = 2.0


# ============================================================
# RIESZ / MONOGENIC PHASE
# ============================================================

def monogenic_phase(image):
    """
    Compute a 2-D Riesz/monogenic phase representation.

    Returns:
        phase
        amplitude
    """

    image = image.astype(np.float64)

    h, w = image.shape

    # Remove DC
    image = image - np.mean(image)

    F = np.fft.fft2(image)

    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]

    kx_grid = np.broadcast_to(kx, (h, w))
    ky_grid = np.broadcast_to(ky, (h, w))

    magnitude = np.sqrt(
        kx_grid ** 2 +
        ky_grid ** 2
    )

    magnitude[0, 0] = 1.0

    # Riesz transforms
    R1 = np.real(
        np.fft.ifft2(
            F * (-1j * kx_grid / magnitude)
        )
    )

    R2 = np.real(
        np.fft.ifft2(
            F * (-1j * ky_grid / magnitude)
        )
    )

    # Local amplitude
    amplitude = np.sqrt(
        image ** 2 +
        R1 ** 2 +
        R2 ** 2
    )

    # Monogenic phase
    phase = np.arctan2(
        np.sqrt(R1 ** 2 + R2 ** 2),
        image
    )

    return phase, amplitude


# ============================================================
# VIDEO OPEN
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
print("PHASE-BASED 440 Hz ROLLING-SHUTTER TEST")
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

print(f"ROI        : x={x1}:{x2}")


# ============================================================
# FIRST FRAME = REFERENCE
# ============================================================

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read first frame")

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
).astype(np.float64)

reference = gray[:, x1:x2]

reference = cv2.GaussianBlur(
    reference,
    (0, 0),
    SIGMA
)

ref_phase, ref_amp = monogenic_phase(
    reference
)


# ============================================================
# EXTRACT ROW PHASE SIGNALS
# ============================================================

phase_rows = []

print("\nExtracting phase...")


for frame_idx in range(nframes):

    if frame_idx == 0:

        phase = ref_phase
        amplitude = ref_amp

    else:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float64)

        image = gray[:, x1:x2]

        image = cv2.GaussianBlur(
            image,
            (0, 0),
            SIGMA
        )

        phase, amplitude = monogenic_phase(
            image
        )


    # --------------------------------------------------------
    # Phase difference relative to reference
    # --------------------------------------------------------

    delta = (
        phase -
        ref_phase
    )

    # Wrap to [-pi, pi]
    delta = np.angle(
        np.exp(1j * delta)
    )


    # --------------------------------------------------------
    # Amplitude weighting
    # --------------------------------------------------------

    weight = amplitude ** 2

    # Prevent extremely weak regions dominating
    weight += 1e-12


    # --------------------------------------------------------
    # Spatially average phase for every row
    # --------------------------------------------------------

    row_phase = np.zeros(height)

    for r in range(height):

        w = weight[r]
        d = delta[r]

        z = np.sum(
            w * np.exp(1j * d)
        ) / np.sum(w)

        row_phase[r] = np.angle(z)


    phase_rows.append(row_phase)

    if frame_idx % 50 == 0:

        print(
            f"Processed frame "
            f"{frame_idx}/{nframes}"
        )


cap.release()

phase_rows = np.asarray(
    phase_rows
)

print(
    f"\nPhase matrix : "
    f"{phase_rows.shape}"
)


# ============================================================
# REMOVE STATIC ROW BIAS
# ============================================================

phase_rows -= np.mean(
    phase_rows,
    axis=0,
    keepdims=True
)


# ============================================================
# VALID ROWS
# ============================================================

row_std = np.std(
    phase_rows,
    axis=0
)

threshold = np.percentile(
    row_std,
    30
)

valid = row_std > threshold

print(
    f"Valid rows  : "
    f"{np.sum(valid)} / {height}"
)


# ============================================================
# CREATE ROW BINS
# ============================================================

row_positions = []
row_phase_signal = []

for start in range(
    0,
    height - ROW_BIN + 1,
    ROW_BIN
):

    end = start + ROW_BIN

    indices = np.arange(
        start,
        end
    )

    valid_here = valid[indices]

    if not np.any(valid_here):
        continue

    row_positions.append(
        np.mean(indices)
    )

    row_phase_signal.append(
        np.mean(
            phase_rows[:, indices],
            axis=1
        )
    )


row_positions = np.asarray(
    row_positions
)

row_phase_signal = np.asarray(
    row_phase_signal
)

print(
    f"Row bins    : "
    f"{len(row_positions)}"
)


# ============================================================
# BUILD ROLLING-SHUTTER TIMESTAMPS
# ============================================================

frame_times = (
    np.arange(nframes) / fps
)[:, None]

row_offsets = (
    row_positions * ROW_DELAY
)[None, :]

time_matrix = (
    frame_times +
    row_offsets
)

signal_matrix = (
    row_phase_signal.T
)

print(
    f"Time matrix   : "
    f"{time_matrix.shape}"
)

print(
    f"Signal matrix : "
    f"{signal_matrix.shape}"
)


# ============================================================
# FLATTEN
# ============================================================

times = time_matrix.ravel()

signal = signal_matrix.ravel()


# ============================================================
# SORT BY TIME
# ============================================================

order = np.argsort(times)

times = times[order]

signal = signal[order]


# ============================================================
# NORMALIZE
# ============================================================

signal -= np.mean(signal)

std = np.std(signal)

if std > 0:
    signal /= std


print(
    f"Total samples : "
    f"{len(signal)}"
)


# ============================================================
# FREQUENCY FIT
# ============================================================

def fit_frequency(freq):

    omega = 2.0 * np.pi * freq

    c = np.cos(
        omega * times
    )

    s = np.sin(
        omega * times
    )

    A = np.column_stack([
        c,
        s,
        np.ones_like(times)
    ])

    coeff, _, _, _ = np.linalg.lstsq(
        A,
        signal,
        rcond=None
    )

    fitted = A @ coeff

    residual = signal - fitted

    signal_power = np.var(fitted)

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


# ============================================================
# SEARCH
# ============================================================

frequencies = np.arange(
    300,
    601,
    0.25
)

amplitudes = np.zeros(
    len(frequencies)
)

snrs = np.zeros(
    len(frequencies)
)


print("\nSearching phase signal...")

for i, f in enumerate(frequencies):

    amplitudes[i], snrs[i] = (
        fit_frequency(f)
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
print("PHASE RESULT")
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
# TOP CANDIDATES
# ============================================================

top = np.argsort(
    amplitudes
)[::-1][:10]

print("\nTop phase-frequency candidates:")

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
    "Phase-Based Rolling-Shutter 440 Hz Reconstruction"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

output_plot = (
    r"C:\Users\Harkiran\Downloads\Mobile Devices"
    r"\A15_440Hz_phase_reconstruction.png"
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
    r"\A15_440Hz_phase_signal.npz"
)

np.savez(
    output_data,
    time=times,
    signal=signal,
    frequencies=frequencies,
    amplitudes=amplitudes,
    snrs=snrs,
    row_delay=ROW_DELAY,
    fps=fps
)

print("\nSaved:")
print(output_plot)
print(output_data)