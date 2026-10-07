import cv2
import numpy as np


VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_test_clean.mp4"
TARGET_FREQ = 440.0

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open calibration video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n==========================================")
print("ROLLING-SHUTTER READOUT AXIS TEST")
print("==========================================")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.6f}")
print(f"Frames     : {nframes}")
print(f"Target     : {TARGET_FREQ} Hz")


frames = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Central 80% to avoid borders
    x1 = int(width * 0.10)
    x2 = int(width * 0.90)

    y1 = int(height * 0.10)
    y2 = int(height * 0.90)

    gray = gray[y1:y2, x1:x2]

    frames.append(gray.astype(np.float64))

cap.release()

frames = np.asarray(frames)

print(f"Frame array: {frames.shape}")


# ----------------------------------------------------------
# TEST VERTICAL AXIS
# ----------------------------------------------------------

# Average horizontally.
# Each remaining element corresponds to a vertical position.
vertical_signal = np.mean(frames, axis=2)

# Remove temporal mean
vertical_signal -= np.mean(
    vertical_signal,
    axis=0,
    keepdims=True
)


# ----------------------------------------------------------
# TEST HORIZONTAL AXIS
# ----------------------------------------------------------

# Average vertically.
# Each remaining element corresponds to a horizontal position.
horizontal_signal = np.mean(frames, axis=1)

horizontal_signal -= np.mean(
    horizontal_signal,
    axis=0,
    keepdims=True
)


def analyze(signal, name):

    n_frames, n_positions = signal.shape

    t = np.arange(n_frames) / fps

    omega = 2 * np.pi * TARGET_FREQ

    phases = []
    amplitudes = []

    for p in range(n_positions):

        y = signal[:, p]

        A = np.column_stack([
            np.cos(omega * t),
            np.sin(omega * t)
        ])

        coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        a = coeff[0]
        b = coeff[1]

        amplitude = np.sqrt(a*a + b*b)

        phase = np.arctan2(-b, a)

        phases.append(phase)
        amplitudes.append(amplitude)

    phases = np.unwrap(np.asarray(phases))
    amplitudes = np.asarray(amplitudes)

    # Keep stronger positions
    threshold = np.percentile(amplitudes, 50)

    valid = amplitudes > threshold

    positions = np.arange(n_positions)

    positions_valid = positions[valid]
    phases_valid = phases[valid]

    if len(positions_valid) < 10:

        print(f"\n{name}")
        print("Not enough valid positions")
        return

    slope, intercept = np.polyfit(
        positions_valid,
        phases_valid,
        1
    )

    predicted = slope * positions_valid + intercept

    ss_res = np.sum(
        (phases_valid - predicted) ** 2
    )

    ss_tot = np.sum(
        (phases_valid - np.mean(phases_valid)) ** 2
    )

    r2 = 1 - ss_res / ss_tot

    row_delay = abs(
        slope / (2 * np.pi * TARGET_FREQ)
    )

    print("\n------------------------------------------")
    print(name)
    print("------------------------------------------")

    print(f"Positions analyzed : {n_positions}")
    print(f"Valid positions    : {len(positions_valid)}")
    print(f"Phase slope        : {slope:.8e} rad/position")
    print(f"R²                 : {r2:.6f}")
    print(f"Estimated delay    : {row_delay * 1e6:.4f} us/position")


analyze(
    vertical_signal,
    "VERTICAL AXIS"
)

analyze(
    horizontal_signal,
    "HORIZONTAL AXIS"
)