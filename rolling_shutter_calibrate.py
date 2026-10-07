import cv2
import numpy as np
import sys
import os
import math

# Known optical modulation used in the calibration experiment.
KNOWN_FREQ_HZ = 10.0
MAX_FRAMES = 240
NUM_ROW_BINS = 120
MIN_R2 = 0.50


def analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 70)
    print("ROLLING-SHUTTER CALIBRATION")
    print("=" * 70)
    print(f"Video        : {video_path}")
    print(f"Resolution   : {width} x {height}")
    print(f"FPS          : {fps:.6f}")
    print(f"Frames       : {frame_count}")
    print(f"Known flicker: {KNOWN_FREQ_HZ:.2f} Hz")
    print("=" * 70)

    frames = []
    for _ in range(min(frame_count, MAX_FRAMES)):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float64))
    cap.release()

    if len(frames) < 30:
        raise RuntimeError("Too few readable frames.")

    frames = np.asarray(frames)
    print(f"Frames analyzed: {len(frames)}")

    # Use the central 80% of the image width.
    x0 = int(width * 0.10)
    x1 = int(width * 0.90)

    # One brightness value per frame and image row.
    signal = frames[:, :, x0:x1].mean(axis=2)

    # Remove each row's DC brightness.
    signal -= signal.mean(axis=0, keepdims=True)

    # Reference sine/cosine at the known optical modulation frequency.
    t = np.arange(len(frames)) / fps
    omega = 2.0 * np.pi * KNOWN_FREQ_HZ
    sin_ref = np.sin(omega * t)
    cos_ref = np.cos(omega * t)

    rows = []
    amplitudes = []
    phases = []

    edges = np.linspace(0, height, NUM_ROW_BINS + 1)

    for b in range(NUM_ROW_BINS):
        y0 = int(edges[b])
        y1 = max(int(edges[b + 1]), y0 + 1)

        s = signal[:, y0:y1].mean(axis=1)

        c = np.sum(s * cos_ref)
        q = np.sum(s * sin_ref)

        amplitude = (2.0 / len(s)) * np.sqrt(c*c + q*q)

        # s ~= A*cos(omega*t - phase)
        phase = math.atan2(q, c)

        rows.append((y0 + y1 - 1) / 2.0)
        amplitudes.append(amplitude)
        phases.append(phase)

    rows = np.asarray(rows, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    phases = np.asarray(phases, dtype=float)

    # Keep reasonably strong row bands.
    threshold = max(np.percentile(amplitudes, 30), 1e-12)
    valid = amplitudes >= threshold

    r = rows[valid]
    p = np.unwrap(phases[valid])

    if len(r) < 10:
        raise RuntimeError("Not enough valid row bands.")

    A = np.column_stack([r, np.ones_like(r)])
    slope, intercept = np.linalg.lstsq(A, p, rcond=None)[0]

    predicted = slope * r + intercept
    ss_res = np.sum((p - predicted) ** 2)
    ss_tot = np.sum((p - np.mean(p)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    print()
    print("CALIBRATION RESULT")
    print("-" * 70)
    print(f"Valid row bands      : {len(r)}/{len(rows)}")
    print(f"Phase slope          : {slope:.10e} rad/row")
    print(f"Linear-fit R^2       : {r2:.6f}")

    if abs(slope) < 1e-12:
        print("No measurable phase gradient was found.")
        return

    # dphase/drow = 2*pi*f*d
    row_delay = abs(slope) / (2.0 * np.pi * KNOWN_FREQ_HZ)
    row_delay_us = row_delay * 1e6
    row_rate = 1.0 / row_delay
    row_nyquist = row_rate / 2.0
    readout_time = row_delay * (height - 1)

    print(f"Estimated row delay : {row_delay_us:.4f} microseconds/row")
    print(f"Estimated row rate  : {row_rate:.2f} rows/sec")
    print(f"Row-rate Nyquist    : {row_nyquist:.2f} Hz")
    print(f"Estimated readout   : {readout_time*1000:.4f} ms")

    if r2 >= MIN_R2:
        print("STATUS: USABLE CALIBRATION CANDIDATE")
    else:
        print("STATUS: WEAK CALIBRATION -- repeat experiment")
        print(f"Recommended R^2 threshold: {MIN_R2:.2f}")

    out_dir = os.path.dirname(os.path.abspath(video_path))

    np.savez(
        os.path.join(out_dir, "rolling_shutter_calibration.npz"),
        rows=rows,
        amplitudes=amplitudes,
        phases=phases,
        valid=valid,
        phase_slope=slope,
        phase_intercept=intercept,
        r2=r2,
        known_frequency_hz=KNOWN_FREQ_HZ,
        estimated_row_delay_seconds=row_delay,
        estimated_row_rate_hz=row_rate,
        estimated_row_nyquist_hz=row_nyquist,
        estimated_readout_seconds=readout_time,
    )

    np.savetxt(
        os.path.join(out_dir, "rolling_shutter_phase_by_row.csv"),
        np.column_stack([rows, amplitudes, phases, valid.astype(int)]),
        delimiter=",",
        header="row,amplitude,phase_rad,valid",
        comments=""
    )

    print()
    print("Saved:")
    print(os.path.join(out_dir, "rolling_shutter_calibration.npz"))
    print(os.path.join(out_dir, "rolling_shutter_phase_by_row.csv"))
    print()
    print("Do NOT use the estimate for audio reconstruction unless")
    print("the R^2 is reasonably strong and the experiment is repeatable.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python rolling_shutter_calibrate.py "calibration_video.mp4"')
        sys.exit(1)

    analyze(sys.argv[1])
