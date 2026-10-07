import cv2
import numpy as np
from scipy.signal import detrend, find_peaks
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_test_clean.mp4"

# From your 10 Hz rolling-shutter calibration
ROW_DELAY = 8.3763e-6       # seconds / row

EXPECTED_FREQ = 440.0

# Use a central ROI to avoid borders
X1_FRAC = 0.15
X2_FRAC = 0.85

# Start with every 4th row to reduce computation
ROW_STEP = 4

# Number of rows used
MAX_ROWS = 1200


# ============================================================
# READ VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video:\n{VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n==========================================")
print("ROLLING-SHUTTER ROW RECONSTRUCTION")
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
# EXTRACT ROW-BRIGHTNESS SIGNAL
# ============================================================

signals = []
frame_numbers = []

print("\nExtracting row signals...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Central horizontal region
    roi = gray[:, x1:x2].astype(np.float64)

    # Average across width.
    # Every image row now becomes one brightness sample.
    row_signal = np.mean(roi, axis=1)

    signals.append(row_signal)
    frame_numbers.append(len(frame_numbers))

cap.release()

signals = np.asarray(signals)

print(f"Raw matrix : {signals.shape}")


# ============================================================
# REMOVE STATIC IMAGE STRUCTURE
# ============================================================

# Remove temporal mean from each row.
signals = signals - np.mean(signals, axis=0, keepdims=True)

# Normalize each row so very bright/dark rows don't dominate.
row_std = np.std(signals, axis=0)

valid = row_std > np.percentile(row_std, 20)

signals[:, ~valid] = 0

print(f"Valid rows : {np.sum(valid)} / {height}")


# ============================================================
# BUILD NON-UNIFORM ROLLING-SHUTTER TIMESTAMPS
# ============================================================

rows = np.arange(height)

# Select rows
selected_rows = rows[::ROW_STEP]

if len(selected_rows) > MAX_ROWS:
    selected_rows = selected_rows[:MAX_ROWS]

print(f"Rows used  : {len(selected_rows)}")


# Each row is captured later than the previous row.
#
# t = frame_time + row * ROW_DELAY

times = []
values = []

for frame_idx in range(signals.shape[0]):

    frame_time = frame_idx / fps

    row_indices = selected_rows

    t = frame_time + row_indices * ROW_DELAY

    y = signals[frame_idx, row_indices]

    times.append(t)
    values.append(y)

times = np.concatenate(times)
values = np.concatenate(values)


# ============================================================
# SORT BY TIME
# ============================================================

order = np.argsort(times)

times = times[order]
values = values[order]

print(f"Samples    : {len(values)}")

# Remove global mean / trend
values = detrend(values)


# ============================================================
# SINUSOID FITTING
# ============================================================

def fit_frequency(freq):

    omega = 2 * np.pi * freq

    A = np.column_stack([
        np.cos(omega * times),
        np.sin(omega * times),
        np.ones_like(times)
    ])

    coeff, _, _, _ = np.linalg.lstsq(A, values, rcond=None)

    fitted = A @ coeff

    residual = values - fitted

    signal_power = np.var(fitted)
    noise_power = np.var(residual)

    if noise_power <= 0:
        snr = np.inf
    else:
        snr = 10 * np.log10(signal_power / noise_power)

    amplitude = np.sqrt(coeff[0] ** 2 + coeff[1] ** 2)

    return amplitude, snr, fitted


# ============================================================
# FREQUENCY SEARCH
# ============================================================

print("\nSearching around 440 Hz...")

frequencies = np.linspace(300, 600, 1201)

amplitudes = []
snrs = []

for f in frequencies:

    amp, snr, _ = fit_frequency(f)

    amplitudes.append(amp)
    snrs.append(snr)

amplitudes = np.asarray(amplitudes)
snrs = np.asarray(snrs)


# ============================================================
# FIND BEST FREQUENCY
# ============================================================

best_idx = np.argmax(amplitudes)

detected_frequency = frequencies[best_idx]
best_amplitude = amplitudes[best_idx]
best_snr = snrs[best_idx]

print("\n==========================================")
print("RESULT")
print("==========================================")

print(f"Expected frequency : {EXPECTED_FREQ:.2f} Hz")
print(f"Detected frequency : {detected_frequency:.2f} Hz")
print(f"Frequency error     : {abs(detected_frequency - EXPECTED_FREQ):.2f} Hz")
print(f"Amplitude           : {best_amplitude:.6e}")

# Calculate SNR specifically at 440 Hz
amp_440, snr_440, fitted_440 = fit_frequency(EXPECTED_FREQ)

print(f"SNR at 440 Hz       : {snr_440:.2f} dB")


# ============================================================
# PLOT FREQUENCY RESPONSE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(frequencies, amplitudes)

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
plt.ylabel("Fitted amplitude")
plt.title("Rolling-Shutter Reconstruction — 440 Hz Test")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_row_reconstruction.png",
    dpi=200
)

plt.show()


# ============================================================
# SAVE NON-UNIFORM SIGNAL
# ============================================================

np.savez(
    r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_row_signal.npz",
    time=times,
    signal=values,
    frequencies=frequencies,
    amplitudes=amplitudes,
    snrs=snrs
)

print("\nSaved:")
print("A15_440Hz_row_reconstruction.png")
print("A15_440Hz_row_signal.npz")