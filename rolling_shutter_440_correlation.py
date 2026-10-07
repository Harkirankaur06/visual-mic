import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_test_clean.mp4"

# Refined calibration from 10 Hz experiment
ROW_DELAY = 7.7435e-6

EXPECTED_FREQ = 440.0

# Central ROI
X1_FRAC = 0.15
X2_FRAC = 0.85

# Small row averaging
ROW_BIN = 2


# ============================================================
# READ VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n==========================================")
print("COHERENT 440 Hz ROLLING-SHUTTER TEST")
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
# EXTRACT ROW BRIGHTNESS
# ============================================================

frames = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float64)

    roi = gray[:, x1:x2]

    # Brightness for each image row
    row_signal = np.mean(
        roi,
        axis=1
    )

    frames.append(row_signal)

cap.release()

frames = np.asarray(frames)

print(f"Raw matrix : {frames.shape}")


# ============================================================
# REMOVE STATIC COMPONENT
# ============================================================

frames -= np.mean(
    frames,
    axis=0,
    keepdims=True
)


# ============================================================
# VALID ROWS
# ============================================================

row_std = np.std(
    frames,
    axis=0
)

threshold = np.percentile(
    row_std,
    30
)

valid = row_std > threshold

print(
    f"Valid rows : "
    f"{np.sum(valid)} / {height}"
)


# ============================================================
# BUILD ROW BINS
# ============================================================

row_positions = []
row_values = []

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

    if not np.any(valid[indices]):
        continue

    row_positions.append(
        np.mean(indices)
    )

    row_values.append(
        np.mean(
            frames[:, indices],
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
    f"Row bins   : "
    f"{len(row_positions)}"
)


# ============================================================
# BUILD CORRECT TIME / SIGNAL ARRAYS
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
    row_values.T
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
    f"Samples       : "
    f"{len(signal)}"
)


# ============================================================
# COHERENT CORRELATION
# ============================================================

def coherent_measure(freq):

    omega = 2 * np.pi * freq

    c = np.cos(
        omega * times
    )

    s = np.sin(
        omega * times
    )

    # Normalize basis functions
    c_norm = np.sqrt(
        np.sum(c * c)
    )

    s_norm = np.sqrt(
        np.sum(s * s)
    )

    c = c / c_norm
    s = s / s_norm

    # Correlation coefficients
    I = np.sum(
        signal * c
    )

    Q = np.sum(
        signal * s
    )

    magnitude = np.sqrt(
        I * I + Q * Q
    )

    return magnitude, I, Q


# ============================================================
# 440 Hz
# ============================================================

m440, I440, Q440 = coherent_measure(
    EXPECTED_FREQ
)


print("\n==========================================")
print("440 Hz COHERENCE")
print("==========================================")

print(
    f"I component : {I440:.8e}"
)

print(
    f"Q component : {Q440:.8e}"
)

print(
    f"Magnitude    : {m440:.8e}"
)


# ============================================================
# FREQUENCY SEARCH
# ============================================================

frequencies = np.arange(
    300,
    601,
    0.25
)

coherence = np.zeros(
    len(frequencies)
)

for i, f in enumerate(frequencies):

    coherence[i], _, _ = (
        coherent_measure(f)
    )


# ============================================================
# BEST FREQUENCY
# ============================================================

best = np.argmax(
    coherence
)

detected = frequencies[best]

print("\n==========================================")
print("COHERENT FREQUENCY SEARCH")
print("==========================================")

print(
    f"Expected : "
    f"{EXPECTED_FREQ:.2f} Hz"
)

print(
    f"Detected : "
    f"{detected:.2f} Hz"
)

print(
    f"Error    : "
    f"{abs(detected - EXPECTED_FREQ):.2f} Hz"
)

print(
    f"440 Hz coherence : "
    f"{m440:.8e}"
)


# ============================================================
# NORMALIZED 440 Hz SCORE
# ============================================================

median_coherence = np.median(
    coherence
)

if median_coherence > 0:

    ratio = (
        m440 /
        median_coherence
    )

    print(
        f"440 Hz / median : "
        f"{ratio:.4f}"
    )


# ============================================================
# TOP CANDIDATES
# ============================================================

top = np.argsort(
    coherence
)[::-1][:10]

print("\nTop coherent frequencies:")

for idx in top:

    print(
        f"{frequencies[idx]:8.2f} Hz   "
        f"{coherence[idx]:.8e}"
    )


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    frequencies,
    coherence
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
    "Coherent correlation magnitude"
)

plt.title(
    "Rolling-Shutter 440 Hz Coherence Test"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

output = (
    r"C:\Users\Harkiran\Downloads\Mobile Devices"
    r"\A15_440Hz_coherence.png"
)

plt.savefig(
    output,
    dpi=200
)

plt.show()

print("\nSaved:")
print(output)