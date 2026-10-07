import cv2
import numpy as np
from scipy.signal import welch
import os
import csv

# ============================================================
# SAMSUNG A15 ROLLING-SHUTTER FREQUENCY RESPONSE TEST
# ============================================================

VIDEO_TESTS = [
    (
        "10 Hz",
        10.0,
        r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_RS_calibration.mp4"
    ),
    (
        "50 Hz",
        50.0,
        r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_RS_50Hz.mp4"
    ),
    (
        "100 Hz",
        100.0,
        r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_RS_100Hz.mp4"
    ),
    (
        "200 Hz",
        200.0,
        r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_RS_200Hz.mp4"
    ),
]

# Use the same calibrated row delay
ROW_DELAY = 8.3763e-6

# Number of horizontal bands
NUM_BANDS = 120

# Analyze central portion of the image
TOP = 0.10
BOTTOM = 0.90
LEFT = 0.20
RIGHT = 0.80


# ============================================================
# ANALYZE ONE VIDEO
# ============================================================

def analyze_video(label, target_freq, video_path):

    print()
    print("=" * 65)
    print(f"ANALYZING {label}")
    print("=" * 65)

    if not os.path.exists(video_path):

        print("ERROR: File not found")
        print(video_path)

        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("ERROR: Could not open video")

        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"Video      : {video_path}")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.6f}")
    print(f"Frames     : {frames}")
    print(f"Target     : {target_freq:.2f} Hz")

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    y_start = int(height * TOP)
    y_end = int(height * BOTTOM)

    x_start = int(width * LEFT)
    x_end = int(width * RIGHT)

    row_edges = np.linspace(
        y_start,
        y_end,
        NUM_BANDS + 1
    ).astype(int)

    # --------------------------------------------------------
    # Read frames
    # --------------------------------------------------------

    row_signals = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float64)

        frame_rows = []

        for i in range(NUM_BANDS):

            ya = row_edges[i]
            yb = row_edges[i + 1]

            region = gray[
                ya:yb,
                x_start:x_end
            ]

            frame_rows.append(
                np.mean(region)
            )

        row_signals.append(frame_rows)

    cap.release()

    row_signals = np.asarray(row_signals)

    print(
        f"Signal matrix: "
        f"{row_signals.shape}"
    )

    if len(row_signals) < 32:

        print("Not enough frames.")

        return None

    # --------------------------------------------------------
    # Remove temporal DC from every row
    # --------------------------------------------------------

    row_signals -= np.mean(
        row_signals,
        axis=0,
        keepdims=True
    )

    # --------------------------------------------------------
    # Compute temporal spectrum for each row
    # --------------------------------------------------------

    row_amplitudes = []

    for r in range(NUM_BANDS):

        signal = row_signals[:, r]

        freqs, power = welch(
            signal,
            fs=fps,
            nperseg=min(
                256,
                len(signal)
            )
        )

        # The normal frame-rate spectrum can only
        # reach fps/2.
        if target_freq <= fps / 2:

            idx = np.argmin(
                np.abs(
                    freqs - target_freq
                )
            )

            row_amplitudes.append(
                np.sqrt(
                    power[idx]
                )
            )

        else:

            # For frequencies above frame-rate Nyquist,
            # the frame-domain signal aliases.
            #
            # We therefore do NOT pretend that the
            # ordinary frame spectrum represents the
            # original modulation.
            row_amplitudes.append(
                np.nan
            )

    row_amplitudes = np.asarray(
        row_amplitudes
    )

    # --------------------------------------------------------
    # Rolling-shutter row-phase analysis
    # --------------------------------------------------------
    #
    # For a known modulation:
    #
    # phase(row) = 2*pi*f*row*ROW_DELAY + constant
    #
    # We estimate phase progression using the
    # frame sequence projected onto the known frequency.
    #
    # --------------------------------------------------------

    valid_rows = []

    phases = []

    amplitudes = []

    t = np.arange(
        len(row_signals)
    ) / fps

    omega = 2.0 * np.pi * target_freq

    cos_ref = np.cos(
        omega * t
    )

    sin_ref = np.sin(
        omega * t
    )

    for r in range(NUM_BANDS):

        signal = row_signals[:, r]

        # Projection onto known modulation.
        c = np.sum(
            signal * cos_ref
        )

        s = np.sum(
            signal * sin_ref
        )

        amplitude = (
            2.0
            * np.sqrt(
                c * c + s * s
            )
            / len(signal)
        )

        phase = np.arctan2(
            -s,
            c
        )

        amplitudes.append(
            amplitude
        )

        phases.append(
            phase
        )

    phases = np.unwrap(
        np.asarray(phases)
    )

    amplitudes = np.asarray(
        amplitudes
    )

    row_centers = (
        row_edges[:-1]
        + row_edges[1:]
    ) / 2.0

    # --------------------------------------------------------
    # Select reliable rows
    # --------------------------------------------------------

    amplitude_threshold = (
        np.percentile(
            amplitudes,
            50
        )
    )

    valid = (
        np.isfinite(phases)
        &
        np.isfinite(amplitudes)
        &
        (
            amplitudes
            >= amplitude_threshold
        )
    )

    x = row_centers[valid]
    y = phases[valid]

    if len(x) < 10:

        print(
            "Not enough valid rows "
            "for phase fitting."
        )

        return None

    # --------------------------------------------------------
    # Linear phase fit
    # --------------------------------------------------------

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    predicted = (
        slope * x
        + intercept
    )

    ss_res = np.sum(
        (y - predicted) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:

        r2 = 1.0 - (
            ss_res / ss_tot
        )

    else:

        r2 = np.nan

    # --------------------------------------------------------
    # Estimate row delay
    # --------------------------------------------------------

    estimated_delay = (
        abs(slope)
        / (
            2.0
            * np.pi
            * target_freq
        )
    )

    # --------------------------------------------------------
    # Expected phase slope from calibration
    # --------------------------------------------------------

    expected_slope = (
        2.0
        * np.pi
        * target_freq
        * ROW_DELAY
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("RESULT")

    print(
        f"Valid rows          : "
        f"{len(x)}/{NUM_BANDS}"
    )

    print(
        f"Measured phase slope: "
        f"{slope:.8e} rad/row"
    )

    print(
        f"Expected slope      : "
        f"{expected_slope:.8e} rad/row"
    )

    print(
        f"Phase-fit R²        : "
        f"{r2:.6f}"
    )

    print(
        f"Estimated row delay : "
        f"{estimated_delay * 1e6:.4f} us"
    )

    print(
        f"Calibrated delay    : "
        f"{ROW_DELAY * 1e6:.4f} us"
    )

    # --------------------------------------------------------
    # Difference from calibration
    # --------------------------------------------------------

    delay_error = (
        (
            estimated_delay
            - ROW_DELAY
        )
        / ROW_DELAY
        * 100.0
    )

    print(
        f"Delay difference    : "
        f"{delay_error:.2f}%"
    )

    return {
        "label": label,
        "target_hz": target_freq,
        "fps": fps,
        "frames": frames,
        "phase_slope": slope,
        "expected_slope": expected_slope,
        "r2": r2,
        "estimated_delay_us":
            estimated_delay * 1e6,
        "calibrated_delay_us":
            ROW_DELAY * 1e6,
        "delay_error_percent":
            delay_error,
    }


# ============================================================
# RUN ALL TESTS
# ============================================================

results = []

for label, freq, path in VIDEO_TESTS:

    result = analyze_video(
        label,
        freq,
        path
    )

    if result is not None:

        results.append(result)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("CAMERA FREQUENCY-RESPONSE SUMMARY")
print("=" * 80)

print(
    f"{'Test':<10}"
    f"{'Target':>10}"
    f"{'R²':>12}"
    f"{'Est.Delay':>15}"
    f"{'Error':>12}"
)

print("-" * 80)

for r in results:

    print(
        f"{r['label']:<10}"
        f"{r['target_hz']:>10.1f}"
        f"{r['r2']:>12.4f}"
        f"{r['estimated_delay_us']:>15.4f}"
        f"{r['delay_error_percent']:>11.2f}%"
    )


# ============================================================
# SAVE CSV
# ============================================================

output_path = (
    r"C:\Users\Harkiran\Downloads"
    r"\Mobile Devices"
    r"\A15_frequency_response_results.csv"
)

if results:

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print("Saved results:")
    print(output_path)

print()
print("DONE.")