
import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt


def amplify_video(
    input_path,
    output_path="amplified_for_audio.mp4",
    alpha=20.0,
    low_hz=0.5,
    high_hz=None,
    levels=3,
):
    """
    Broadband temporal motion amplification.

    This replaces the original 0.8-2.5 Hz restriction with a band that
    extends toward the video's Nyquist frequency. That preserves as much
    temporal motion as the camera actually sampled.

    IMPORTANT:
    Normal video still cannot recover frequencies above FPS/2.
    This code amplifies motion; it does not create missing audio samples.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    nyquist = fps / 2.0

    if high_hz is None:
        high_hz = 0.90 * nyquist

    high_hz = min(high_hz, 0.95 * nyquist)

    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError(
            f"Invalid temporal band {low_hz}-{high_hz} Hz "
            f"for {fps:.3f} FPS video."
        )

    print(f"FPS:       {fps:.4f}")
    print(f"Nyquist:   {nyquist:.4f} Hz")
    print(f"Band:      {low_hz:.4f} - {high_hz:.4f} Hz")
    print(f"Alpha:     {alpha}")

    # Offline zero-phase temporal filter.
    sos = butter(
        2,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fps,
        output="sos",
    )

    # Read the video.
    luminance = []
    chroma = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        y = ycrcb[:, :, 0].astype(np.float32) / 255.0
        luminance.append(y)
        chroma.append(ycrcb[:, :, 1:3].copy())

    cap.release()

    n = len(luminance)

    if n < 12:
        raise RuntimeError(
            f"Only {n} frames found. Use a longer video."
        )

    print(f"Frames:    {n}")
    print(f"Duration:  {n / fps:.3f} seconds")

    video = np.stack(luminance, axis=0)

    # Build Laplacian pyramids.
    pyramids = []

    for i in range(n):
        current = video[i]
        pyr = []

        for _ in range(levels):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(
                down,
                dstsize=(current.shape[1], current.shape[0]),
            )
            pyr.append(current - up)
            current = down

        pyr.append(current)
        pyramids.append(pyr)

    # Amplify each spatial band across time.
    amplified_pyramid = []

    for level in range(levels):
        stack = np.stack(
            [pyramids[t][level] for t in range(n)],
            axis=0,
        )

        motion = sosfiltfilt(sos, stack, axis=0)

        # Add only the filtered motion back to the original.
        amplified = stack + alpha * motion
        amplified_pyramid.append(amplified)

        print(f"Amplified pyramid level {level + 1}/{levels}")

    # Keep the low-frequency residual unchanged.
    amplified_pyramid.append(
        np.stack(
            [pyramids[t][-1] for t in range(n)],
            axis=0,
        )
    )

    # Reconstruct frames.
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create: {output_path}")

    for t in range(n):
        current = amplified_pyramid[-1][t]

        for level in reversed(range(levels)):
            current = cv2.pyrUp(
                current,
                dstsize=(
                    amplified_pyramid[level][t].shape[1],
                    amplified_pyramid[level][t].shape[0],
                ),
            )
            current += amplified_pyramid[level][t]

        y8 = np.clip(current * 255.0, 0, 255).astype(np.uint8)

        ycrcb = np.empty((height, width, 3), dtype=np.uint8)
        ycrcb[:, :, 0] = y8
        ycrcb[:, :, 1:3] = chroma[t]

        out_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        writer.write(out_frame)

        if (t + 1) % 10 == 0 or t == n - 1:
            print(f"Written {t + 1}/{n} frames")

    writer.release()

    print()
    print(f"Saved: {output_path}")
    print(f"Output duration: {n / fps:.3f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Broadband temporal motion amplification"
    )

    parser.add_argument("input", help="Input video")
    parser.add_argument(
        "--output",
        default="amplified_for_audio.mp4",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--low",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--high",
        type=float,
        default=None,
        help="Default: 90%% of Nyquist",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    amplify_video(
        args.input,
        args.output,
        alpha=args.alpha,
        low_hz=args.low,
        high_hz=args.high,
        levels=args.levels,
    )
