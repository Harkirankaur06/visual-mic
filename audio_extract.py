
import argparse
import wave
import cv2
import numpy as np
from scipy import signal


def steerable_like_phase_signal(video_path, levels=3, scale=1.0):
    """
    Extracts a temporal phase-motion signal from an amplified video.

    This is the practical Python counterpart of the VMSlim idea:
    spatial band-pass decomposition -> local phase change -> temporal signal.

    IMPORTANT:
    A normal ~30 FPS video only provides ordinary frame-to-frame temporal
    samples up to FPS/2. This function therefore reports that limit instead
    of pretending that 2200 Hz information exists.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0

        if scale != 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        frames.append(gray)

    cap.release()

    if len(frames) < 3:
        raise RuntimeError("Video contains too few frames.")

    video = np.stack(frames)
    n, h, w = video.shape

    print(f"Frames:      {n}")
    print(f"FPS:         {fps:.6f}")
    print(f"Duration:    {n / fps:.4f} s")
    print(f"Nyquist:     {fps / 2.0:.6f} Hz")
    print(f"Resolution:  {w} x {h}")

    # Use spatial Gaussian/Laplacian bands. We extract phase-like motion
    # from the analytic signal of each band in time.
    signals = []
    weights = []

    current = video

    for level in range(levels):
        # Spatial low-pass for the next pyramid level.
        low = np.empty_like(current)

        for t in range(n):
            low[t] = cv2.GaussianBlur(
                current[t],
                (0, 0),
                sigmaX=1.2,
                sigmaY=1.2,
            )

        band = current - low

        # Temporal mean of the band gives a robust global motion trace.
        spatial_trace = band.reshape(n, -1).mean(axis=1)

        # Remove DC.
        spatial_trace -= np.mean(spatial_trace)

        # Analytic temporal representation.
        analytic = signal.hilbert(spatial_trace)

        phase = np.unwrap(np.angle(analytic))

        # Frame-to-frame phase change.
        dphase = np.diff(phase, prepend=phase[0])

        # Robustly remove the mean.
        dphase -= np.mean(dphase)

        # Weight each band by its temporal energy.
        weight = np.std(spatial_trace) + 1e-12

        signals.append(dphase)
        weights.append(weight)

        print(
            f"Spatial level {level + 1}/{levels}: "
            f"motion std={weight:.6e}"
        )

        current = low[:, ::2, ::2]

        if current.shape[1] < 8 or current.shape[2] < 8:
            break

    weights = np.asarray(weights)
    weights /= np.sum(weights)

    recovered = np.zeros(n, dtype=np.float64)

    for s, w in zip(signals, weights):
        recovered += w * s

    # Remove slow drift and normalize.
    recovered -= signal.savgol_filter(
        recovered,
        window_length=min(
            n if n % 2 else n - 1,
            max(5, int(round(fps * 0.5)) | 1),
        ),
        polyorder=2,
    )

    recovered = recovered.astype(np.float64)

    # Scale safely to avoid clipping.
    peak = np.max(np.abs(recovered)) + 1e-12
    recovered /= peak

    return recovered, fps


def write_wav(path, samples, sample_rate):
    """
    Write a mono 16-bit WAV without resampling the underlying information.
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = np.clip(samples, -1.0, 1.0)

    pcm = (samples * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(round(sample_rate)))
        wf.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="VMSlim-style phase motion extraction"
    )

    parser.add_argument(
        "video",
        help="Amplified input video",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output WAV",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Spatial resize factor before extraction",
    )

    args = parser.parse_args()

    if args.output is None:
        if args.video.lower().endswith(".mp4"):
            output = args.video[:-4] + "_recovered.wav"
        else:
            output = args.video + "_recovered.wav"
    else:
        output = args.output

    samples, fps = steerable_like_phase_signal(
        args.video,
        levels=args.levels,
        scale=args.scale,
    )

    # The signal genuinely contains one sample per video frame.
    # Therefore its honest sample rate is the video FPS.
    write_wav(output, samples, fps)

    print()
    print("Extraction complete.")
    print(f"WAV:         {output}")
    print(f"Samples:     {len(samples)}")
    print(f"Sample rate: {fps:.6f} Hz")
    print(f"Duration:    {len(samples) / fps:.4f} s")
    print()
    print(
        "NOTE: This WAV contains the frame-rate phase-motion signal. "
        "It is NOT artificially upsampled to 2200 Hz."
    )


if __name__ == "__main__":
    main()
