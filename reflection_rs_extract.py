import cv2
import numpy as np
from scipy import signal
import os

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\light vibe exp"

os.makedirs(OUTDIR, exist_ok=True)

# Tight reflected-light region
x1, x2 = 180, 320
y1, y2 = 760, 940

ROW_DELAY = 7.7435e-6
FPS = 30.0
TARGET_FS = 48000

cap = cv2.VideoCapture(VIDEO)

frames = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[y1:y2, x1:x2]

    # Mean intensity across the width for each row
    row_signal = np.mean(roi.astype(np.float64), axis=1)

    frames.append(row_signal)

cap.release()

X = np.asarray(frames)

print("Frames:", X.shape[0])
print("Rows:", X.shape[1])
print("ROI:", x1, x2, y1, y2)

# ---------------------------------------------------------
# Remove spatial DC / slow illumination variation
# ---------------------------------------------------------

X = X - np.mean(X, axis=0, keepdims=True)

# Remove frame-wise global brightness changes
X = X - np.mean(X, axis=1, keepdims=True)

# ---------------------------------------------------------
# Rolling-shutter reconstruction
# ---------------------------------------------------------

n_frames, n_rows = X.shape

times = (
    np.arange(n_frames)[:, None] / FPS
    + np.arange(n_rows)[None, :] * ROW_DELAY
)

times = times.ravel()
values = X.ravel()

order = np.argsort(times)

times = times[order]
values = values[order]

# Remove duplicate/near-duplicate timestamps
dt = np.median(np.diff(times))
print("Median RS time step:", dt)
print("Equivalent sample rate:", 1.0 / dt)

# Uniform reconstruction
t_uniform = np.arange(
    times[0],
    times[-1],
    1.0 / TARGET_FS
)

audio = np.interp(
    t_uniform,
    times,
    values
)

# ---------------------------------------------------------
# Bandpass candidates
# ---------------------------------------------------------

def bandpass(x, low, high):
    nyq = TARGET_FS / 2
    sos = signal.butter(
        5,
        [low / nyq, high / nyq],
        btype="bandpass",
        output="sos"
    )
    return signal.sosfiltfilt(sos, x)

# Normalize helper
def normalize(x):
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return (x * 0.95 * 32767).astype(np.int16)

# Candidate 1: speech-focused
speech = bandpass(audio, 150, 5000)

# Candidate 2: lower speech
speech_low = bandpass(audio, 80, 2500)

# Candidate 3: wider
wide = bandpass(audio, 80, 8000)

# ---------------------------------------------------------
# Save WAV
# ---------------------------------------------------------

from scipy.io import wavfile

wavfile.write(
    os.path.join(OUTDIR, "reflection_rs_speech.wav"),
    TARGET_FS,
    normalize(speech)
)

wavfile.write(
    os.path.join(OUTDIR, "reflection_rs_speech_low.wav"),
    TARGET_FS,
    normalize(speech_low)
)

wavfile.write(
    os.path.join(OUTDIR, "reflection_rs_wide.wav"),
    TARGET_FS,
    normalize(wide)
)

# Save raw reconstruction too
np.savez_compressed(
    os.path.join(OUTDIR, "reflection_rs_reconstruction.npz"),
    audio=audio,
    fs=TARGET_FS,
    row_delay=ROW_DELAY,
    roi=np.array([x1,x2,y1,y2])
)

print()
print("DONE")
print("Saved to:", OUTDIR)
print()
print("WAV files:")
print("reflection_rs_speech.wav")
print("reflection_rs_speech_low.wav")
print("reflection_rs_wide.wav")
