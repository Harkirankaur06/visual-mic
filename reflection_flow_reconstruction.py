import cv2
import numpy as np
import os
from scipy import signal
from scipy.io import wavfile

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\light vibe exp"

os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------
# Reflected-light region
# ---------------------------------------------------------
x1, x2 = 180, 320
y1, y2 = 760, 940

FPS = 30.0
ROW_DELAY = 7.7435e-6
TARGET_FS = 48000

cap = cv2.VideoCapture(VIDEO)

ok, first = cap.read()
if not ok:
    raise RuntimeError("Could not open video")

prev = cv2.cvtColor(
    first[y1:y2, x1:x2],
    cv2.COLOR_BGR2GRAY
).astype(np.float32)

# Slight blur makes optical flow less sensitive to camera noise
prev = cv2.GaussianBlur(prev, (5, 5), 0)

all_x = []
all_y = []
all_mag = []

frame_count = 1

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(
        frame[y1:y2, x1:x2],
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    flow = cv2.calcOpticalFlowFarneback(
        prev,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=25,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0
    )

    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx**2 + fy**2)

    # Spatial gradient of the previous frame.
    # This tells us where actual image structure exists.
    gx = cv2.Sobel(prev, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prev, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(gx**2 + gy**2)

    # Ignore extremely weak texture and extreme flow outliers.
    threshold = np.percentile(grad, 55)

    valid = (
        (grad > threshold) &
        np.isfinite(fx) &
        np.isfinite(fy) &
        (mag < 5.0)
    )

    # Robust weighted row averages.
    row_x = np.zeros(fx.shape[0])
    row_y = np.zeros(fy.shape[0])
    row_mag = np.zeros(fx.shape[0])

    for r in range(fx.shape[0]):
        m = valid[r]

        if np.count_nonzero(m) < 5:
            continue

        w = grad[r][m] + 1e-6

        row_x[r] = np.sum(fx[r][m] * w) / np.sum(w)
        row_y[r] = np.sum(fy[r][m] * w) / np.sum(w)
        row_mag[r] = np.sum(mag[r][m] * w) / np.sum(w)

    all_x.append(row_x)
    all_y.append(row_y)
    all_mag.append(row_mag)

    prev = gray
    frame_count += 1

cap.release()

X = np.asarray(all_x)
Y = np.asarray(all_y)
M = np.asarray(all_mag)

print("Flow matrices:", X.shape)
print("Frames processed:", frame_count)

# ---------------------------------------------------------
# Remove per-frame global camera drift
# ---------------------------------------------------------
# Median across rows removes common motion affecting the
# entire ROI while preserving spatially varying motion.

X = X - np.median(X, axis=1, keepdims=True)
Y = Y - np.median(Y, axis=1, keepdims=True)

# ---------------------------------------------------------
# Build rolling-shutter timestamps
# ---------------------------------------------------------

n_frames, n_rows = X.shape

frame_times = np.arange(n_frames) / FPS
row_times = np.arange(n_rows) * ROW_DELAY

T = (
    frame_times[:, None]
    + row_times[None, :]
).ravel()

signals = {
    "flow_x": X.ravel(),
    "flow_y": Y.ravel(),
    "flow_mag": M.ravel()
}

# Sort according to rolling-shutter time
order = np.argsort(T)
T = T[order]

# ---------------------------------------------------------
# Uniform resampling
# ---------------------------------------------------------

t_uniform = np.arange(
    T[0],
    T[-1],
    1.0 / TARGET_FS
)

def reconstruct(values):

    values = values[order]

    # Remove slow drift before interpolation
    values = signal.detrend(values, type="linear")

    return np.interp(
        t_uniform,
        T,
        values
    )

def bandpass_audio(x):

    # Broad range for music rather than speech
    sos = signal.butter(
        5,
        [20/(TARGET_FS/2), 8000/(TARGET_FS/2)],
        btype="bandpass",
        output="sos"
    )

    x = signal.sosfiltfilt(sos, x)

    # Remove residual DC
    x -= np.mean(x)

    return x

def normalize(x):

    x = x / (np.max(np.abs(x)) + 1e-12)

    return (
        x * 0.95 * 32767
    ).astype(np.int16)

# ---------------------------------------------------------
# Generate candidates
# ---------------------------------------------------------

for name, values in signals.items():

    audio = reconstruct(values)

    audio = bandpass_audio(audio)

    path = os.path.join(
        OUTDIR,
        "reflection_flow_" + name + ".wav"
    )

    wavfile.write(
        path,
        TARGET_FS,
        normalize(audio)
    )

    print("Saved:", path)

# ---------------------------------------------------------
# Save numerical data
# ---------------------------------------------------------

np.savez_compressed(
    os.path.join(
        OUTDIR,
        "reflection_flow_reconstruction.npz"
    ),
    flow_x=X,
    flow_y=Y,
    flow_magnitude=M,
    timestamps=T,
    uniform_time=t_uniform,
    row_delay=ROW_DELAY,
    fps=FPS,
    roi=np.array([x1, x2, y1, y2])
)

print()
print("DONE")
print()
print("Listen to these:")
print("1. reflection_flow_flow_x.wav")
print("2. reflection_flow_flow_y.wav")
print("3. reflection_flow_flow_mag.wav")
