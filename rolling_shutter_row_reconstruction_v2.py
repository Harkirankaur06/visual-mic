import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_test_clean.mp4"

# Refined 10 Hz calibration
ROW_DELAY = 7.7435e-6       # seconds / row

EXPECTED_FREQ = 440.0

# Use central part of image
X1_FRAC = 0.15
X2_FRAC = 0.85

# Average small groups of rows.
# Do NOT average hundreds of rows together.
ROW_BIN = 2

# Frequency search
F_MIN = 300.0
F_MAX = 600.0
F_STEP = 0.25


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open:\n{VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n==========================================")
print("ROLLING-SHUTTER RECONSTRUCTION V2")
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
# EXTRACT ROW SIGNALS
# ============================================================

all_rows = []

print("\nReading video...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[:, x1:x2].astype(np.float64)

    # Average horizontally.
    # Each image row produces one brightness measurement.
    row_signal = np.mean(roi, axis=1)

    all_rows.append(row_signal)

cap.release()

all_rows = np.asarray(all_rows)

print(f"Raw matrix : {all_rows.shape}")


# ============================================================
# REMOVE STATIC ROW BRIGHTNESS
# ============================================================

# Remove temporal mean of every row.
row_signal = (
    all_rows -
    np.mean(all_rows, axis=0, keepdims=True)
)


# ============================================================
# NORMALIZE ROWS
# ============================================================

row_std = np.std(row_signal, axis=0)

valid_threshold = np.percentile(row_std, 30)

valid_rows = row_std > valid_threshold

print(f"Valid rows : {np.sum(valid_rows)} / {height}")


# ============================================================
# BIN ADJACENT ROWS
# ============================================================

row_positions = []
row_values = []

for start in range(0, height - ROW_BIN + 1, ROW_BIN):

    end = start + ROW_BIN

    rows_here = np.arange(start, end)

    # Only retain bins containing valid rows
    if not np.any(valid_rows[rows_here]):
        continue

    value = np.mean(
        row_signal[:, rows_here],
        axis=1
    )

    position = np.mean(rows_here)

    row_positions.append(position)
    row_values.append(value)


row_positions = np.asarray(row_positions)
row_values = np.asarray(row_values)

print(f"Row bins   : {len(row_positions)}")


# ============================================================
# CONSTRUCT NON-UNIFORM TIME SAMPLES
# ============================================================

# Create the complete frame × row time grid directly.
#
# Shape:
#   frame_times -> (400, 1)
#   row_offsets -> (1, number_of_row_bins)
#   times      -> (400, number_of_row_bins)

frame_times = (
    np.arange(nframes, dtype=np.float64) / fps
)[:, None]

row_offsets = (
    row_positions * ROW_DELAY
)[None, :]

times_matrix = frame_times + row_offsets

# Corresponding brightness values
values_matrix = row_values.T.copy()

# Flatten both arrays in exactly the same order
times = times_matrix.ravel()
values = values_matrix.ravel()

print(f"Time matrix  : {times_matrix.shape}")
print(f"Value matrix : {values_matrix.shape}")


# ============================================================
# SORT TEMPORALLY
# ============================================================

order = np.argsort(times)

times = times[order]
values = values[order]


# ============================================================
# NORMALIZE SIGNAL
# ============================================================

values = values - np.mean(values)

std = np.std(values)

if std > 0:
    values = values / std


print(f"Total samples : {len(values)}")

print(
    f"Time range    : "
    f"{times[0]:.6f} - {times[-1]:.6f} s"
)
# ============================================================
# SINUSOIDAL LEAST-SQUARES MODEL
# ============================================================

def frequency_score(freq):

    omega = 2.0 * np.pi * freq

    c = np.cos(omega * times)
    s = np.sin(omega * times)

    A = np.column_stack([
        c,
        s,
        np.ones_like(times)
    ])

    coeff, _, _, _ = np.linalg.lstsq(
        A,
        values,
        rcond=None
    )

    fitted = A @ coeff

    # Explained power
    signal_power = np.var(fitted)

    residual = values - fitted

    residual_power = np.var(residual)

    if residual_power <= 1e-20:
        snr = np.inf
    else:
        snr = 10.0 * np.log10(
            signal_power / residual_power
        )

    amplitude = np.sqrt(
        coeff[0] ** 2 +
        coeff[1] ** 2
    )

    return amplitude, snr


# ============================================================
# FREQUENCY SEARCH
# ============================================================

frequencies = np.arange(
    F_MIN,
    F_MAX + F_STEP,
    F_STEP
)

amplitudes = np.zeros(len(frequencies))
snrs = np.zeros(len(frequencies))


print("\nSearching frequency...")

for i, f in enumerate(frequencies):

    amplitudes[i], snrs[i] = frequency_score(f)


# ============================================================
# BEST FREQUENCY
# ============================================================

best_index = np.argmax(amplitudes)

detected_frequency = frequencies[best_index]

best_amplitude = amplitudes[best_index]


# Exact 440 Hz measurement
amp_440, snr_440 = frequency_score(
    EXPECTED_FREQ
)


print("\n==========================================")
print("RESULT")
print("==========================================")

print(
    f"Expected frequency : "
    f"{EXPECTED_FREQ:.2f} Hz"
)

print(
    f"Detected frequency : "
    f"{detected_frequency:.2f} Hz"
)

print(
    f"Frequency error     : "
    f"{abs(detected_frequency - EXPECTED_FREQ):.2f} Hz"
)

print(
    f"Amplitude at peak   : "
    f"{best_amplitude:.6e}"
)

print(
    f"SNR at 440 Hz       : "
    f"{snr_440:.2f} dB"
)


# ============================================================
# TOP 10 PEAKS
# ============================================================

top_indices = np.argsort(
    amplitudes
)[::-1][:10]

print("\nTop frequency candidates:")

for idx in top_indices:

    print(
        f"{frequencies[idx]:8.2f} Hz   "
        f"Amplitude={amplitudes[idx]:.6e}   "
        f"SNR={snrs[idx]:7.2f} dB"
    )


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(11, 5))

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
    detected_frequency,
    linestyle=":",
    label=f"Detected {detected_frequency:.2f} Hz"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Least-squares amplitude")

plt.title(
    "Rolling-Shutter Row-wise Frequency Reconstruction"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

OUTPUT_PLOT = (
    r"C:\Users\Harkiran\Downloads\Mobile Devices"
    r"\A15_440Hz_row_reconstruction_v2.png"
)

plt.savefig(
    OUTPUT_PLOT,
    dpi=200
)

plt.show()


# ============================================================
# SAVE DATA
# ============================================================

OUTPUT_DATA = (
    r"C:\Users\Harkiran\Downloads\Mobile Devices"
    r"\A15_440Hz_row_signal_v2.npz"
)

np.savez(
    OUTPUT_DATA,
    time=times,
    signal=values,
    frequencies=frequencies,
    amplitudes=amplitudes,
    snrs=snrs,
    row_delay=ROW_DELAY,
    fps=fps
)

print("\nSaved:")
print(OUTPUT_PLOT)
print(OUTPUT_DATA)