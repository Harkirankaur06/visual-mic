import cv2
import numpy as np
import os
from scipy import signal
from scipy.io import wavfile

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\light vibe exp"

os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------
# REFLECTED-LIGHT ROI
# ---------------------------------------------------------
x1, x2 = 180, 320
y1, y2 = 760, 940

FPS = 30.0
ROW_DELAY = 7.7435e-6
TARGET_FS = 48000

# ---------------------------------------------------------
# Riesz transform
# ---------------------------------------------------------
def riesz_phase(img, sigma):

    img = img.astype(np.float64)

    # Gaussian spatial scale
    low = cv2.GaussianBlur(
        img.astype(np.float32),
        (0, 0),
        sigma
    ).astype(np.float64)

    h, w = low.shape

    F = np.fft.fft2(low)

    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]

    k = np.sqrt(kx**2 + ky**2) + 1e-12

    # Riesz filters
    Rx_filter = -1j * kx / k
    Ry_filter = -1j * ky / k

    rx = np.real(
        np.fft.ifft2(F * Rx_filter)
    )

    ry = np.real(
        np.fft.ifft2(F * Ry_filter)
    )

    riesz_mag = np.sqrt(rx**2 + ry**2)

    # Monogenic phase
    phase = np.arctan2(
        riesz_mag,
        low
    )

    amplitude = np.sqrt(
        low**2 +
        riesz_mag**2
    )

    return phase, amplitude


# ---------------------------------------------------------
# Read video
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO)

frames = []

while True:

    ok, frame = cap.read()

    if not ok:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[y1:y2, x1:x2]

    # Normalize each frame only very lightly.
    roi = roi.astype(np.float64)

    frames.append(roi)

cap.release()

V = np.asarray(frames)

print("Video:", V.shape)
print("ROI:", x1, x2, y1, y2)

# ---------------------------------------------------------
# Multiple spatial scales
# ---------------------------------------------------------

scales = [1.0, 2.0, 4.0]

all_outputs = {}

for sigma in scales:

    print()
    print("Processing scale:", sigma)

    phases = []
    amplitudes = []

    for i in range(len(V)):

        phase, amp = riesz_phase(
            V[i],
            sigma
        )

        phases.append(phase)
        amplitudes.append(amp)

    phases = np.asarray(phases)
    amplitudes = np.asarray(amplitudes)

    # -----------------------------------------------------
    # Temporal phase difference
    # -----------------------------------------------------

    # Complex representation avoids phase wrapping.
    z = np.exp(1j * phases)

    phase_diff = np.angle(
        z[1:] * np.conj(z[:-1])
    )

    amp = amplitudes[1:]

    # -----------------------------------------------------
    # Row-wise robust weighted phase
    # -----------------------------------------------------

    n_frames, n_rows, n_cols = phase_diff.shape

    row_signal = np.zeros(
        (n_frames, n_rows),
        dtype=np.float64
    )

    for f in range(n_frames):

        p = phase_diff[f]
        a = amp[f]

        # Reject very weak phase responses
        threshold = np.percentile(
            a,
            35
        )

        valid = (
            np.isfinite(p) &
            np.isfinite(a) &
            (a > threshold)
        )

        for r in range(n_rows):

            m = valid[r]

            if np.count_nonzero(m) < 5:
                continue

            weights = a[r][m]

            # Robust weighted median-ish estimate:
            # weighted mean after clipping outliers.
            vals = p[r][m]

            lo, hi = np.percentile(
                vals,
                [10, 90]
            )

            keep = (
                (vals >= lo) &
                (vals <= hi)
            )

            vals = vals[keep]
            weights = weights[keep]

            if len(vals) > 0:
                row_signal[f, r] = (
                    np.sum(vals * weights) /
                    (np.sum(weights) + 1e-12)
                )

    # -----------------------------------------------------
    # Remove frame-global phase motion
    # -----------------------------------------------------

    row_signal -= np.median(
        row_signal,
        axis=1,
        keepdims=True
    )

    all_outputs[sigma] = row_signal

    np.savez_compressed(
        os.path.join(
            OUTDIR,
            f"phase_rows_sigma_{sigma}.npz"
        ),
        phase_rows=row_signal
    )

# ---------------------------------------------------------
# Rolling-shutter reconstruction
# ---------------------------------------------------------

def reconstruct(row_signal):

    n_frames, n_rows = row_signal.shape

    # The phase difference corresponds to frame-to-frame
    # displacement, so associate it with frame intervals.
    frame_times = (
        np.arange(n_frames) / FPS
    )

    row_times = (
        np.arange(n_rows) * ROW_DELAY
    )

    T = (
        frame_times[:, None]
        + row_times[None, :]
    ).ravel()

    values = row_signal.ravel()

    order = np.argsort(T)

    T = T[order]
    values = values[order]

    # Remove slow drift
    values = signal.detrend(
        values,
        type="linear"
    )

    # Remove duplicate timestamps if any
    keep = np.concatenate(
        [[True], np.diff(T) > 1e-10]
    )

    T = T[keep]
    values = values[keep]

    t_uniform = np.arange(
        T[0],
        T[-1],
        1.0 / TARGET_FS
    )

    audio = np.interp(
        t_uniform,
        T,
        values
    )

    return audio


# ---------------------------------------------------------
# Audio filters
# ---------------------------------------------------------

def make_audio(x):

    x = x - np.mean(x)

    # Music-oriented bandwidth
    sos = signal.butter(
        5,
        [
            20 / (TARGET_FS / 2),
            8000 / (TARGET_FS / 2)
        ],
        btype="bandpass",
        output="sos"
    )

    x = signal.sosfiltfilt(
        sos,
        x
    )

    x -= np.mean(x)

    return x


def normalize(x):

    peak = np.max(
        np.abs(x)
    )

    if peak < 1e-12:
        return np.zeros(
            len(x),
            dtype=np.int16
        )

    x = x / peak

    return (
        x * 0.95 * 32767
    ).astype(np.int16)


# ---------------------------------------------------------
# Save 3 scale candidates
# ---------------------------------------------------------

for sigma, rows in all_outputs.items():

    print(
        "Reconstructing sigma:",
        sigma
    )

    audio = reconstruct(rows)
    audio = make_audio(audio)

    path = os.path.join(
        OUTDIR,
        f"reflection_phase_sigma_{sigma}.wav"
    )

    wavfile.write(
        path,
        TARGET_FS,
        normalize(audio)
    )

    print("Saved:", path)

print()
print("===================================")
print("DONE")
print("===================================")
print()
print("Listen to:")
print("reflection_phase_sigma_1.0.wav")
print("reflection_phase_sigma_2.0.wav")
print("reflection_phase_sigma_4.0.wav")
