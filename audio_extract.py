
import argparse
import cv2
import numpy as np
from scipy import signal
import soundfile as sf


def extract_and_shift(
    video_path,
    output_path,
    target_hz=400.0,
    levels=3,
    scale=0.5,
):
    """
    Extract a low-frequency visual-motion signal and frequency-shift it
    into the audible range.

    This is an EXPERIMENTAL diagnostic:
    it makes detected low-frequency motion audible. It is NOT claimed
    to reconstruct the original acoustic waveform.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Could not determine FPS.")

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64) / 255.0

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

    if len(frames) < 8:
        raise RuntimeError("Video is too short.")

    video = np.stack(frames)
    n = len(video)

    print(f"Frames:       {n}")
    print(f"FPS:          {fps:.6f}")
    print(f"Duration:     {n / fps:.4f} s")
    print(f"Nyquist:      {fps / 2:.6f} Hz")

    # ------------------------------------------------------------
    # Extract motion at several spatial scales.
    # ------------------------------------------------------------

    traces = []

    current = video

    for level in range(levels):
        blurred = np.empty_like(current)

        for i in range(n):
            blurred[i] = cv2.GaussianBlur(
                current[i],
                (0, 0),
                sigmaX=1.2,
            )

        band = current - blurred

        # Global spatial projection.
        trace = np.mean(band, axis=(1, 2))

        # Remove DC.
        trace -= np.mean(trace)

        traces.append(trace)

        print(
            f"Spatial level {level + 1}: "
            f"std={np.std(trace):.8e}"
        )

        current = blurred[:, ::2, ::2]

        if current.shape[1] < 8 or current.shape[2] < 8:
            break

    # ------------------------------------------------------------
    # Combine spatial bands.
    # ------------------------------------------------------------

    weights = np.array(
        [np.std(x) + 1e-12 for x in traces]
    )

    weights /= np.sum(weights)

    motion = np.zeros(n)

    for x, w in zip(traces, weights):
        motion += w * x

    # Remove DC / very slow drift.
    motion -= np.mean(motion)

    # ------------------------------------------------------------
    # Interpolate the measured signal.
    #
    # IMPORTANT:
    # This does NOT create new information.
    # It only gives us a smooth representation before
    # frequency translation.
    # ------------------------------------------------------------

    t_original = np.arange(n) / fps

    audio_rate = 16000

    duration = (n - 1) / fps

    t_audio = np.arange(
        int(np.ceil(duration * audio_rate)) + 1
    ) / audio_rate

    interpolated = np.interp(
        t_audio,
        t_original,
        motion,
    )

    # ------------------------------------------------------------
    # Frequency-shift the measured motion.
    #
    # We use an analytic signal:
    #
    # x(t) -> Hilbert(x)
    # x_a(t) * exp(j 2*pi*f_shift*t)
    #
    # The real part becomes audible.
    # ------------------------------------------------------------

    analytic = signal.hilbert(interpolated)

    shifted = np.real(
        analytic *
        np.exp(
            2j * np.pi * target_hz * t_audio
        )
    )

    # ------------------------------------------------------------
    # Band-limit the shifted signal around the carrier.
    # ------------------------------------------------------------

    low = max(20.0, target_hz - 100.0)
    high = min(
        audio_rate / 2 - 100.0,
        target_hz + 100.0,
    )

    if high > low:
        sos = signal.butter(
            4,
            [low, high],
            btype="bandpass",
            fs=audio_rate,
            output="sos",
        )

        shifted = signal.sosfiltfilt(
            sos,
            shifted,
        )

    # Normalize safely.
    peak = np.max(np.abs(shifted)) + 1e-12
    shifted /= peak

    # Gentle fade in/out to avoid clicks.
    fade = min(
        int(0.02 * audio_rate),
        len(shifted) // 4,
    )

    if fade > 0:
        ramp = np.linspace(0, 1, fade)
        shifted[:fade] *= ramp
        shifted[-fade:] *= ramp[::-1]

    sf.write(
        output_path,
        shifted.astype(np.float32),
        audio_rate,
        subtype="PCM_16",
    )

    print()
    print("Finished.")
    print(f"WAV:          {output_path}")
    print(f"Audio rate:   {audio_rate} Hz")
    print(f"Carrier:      {target_hz} Hz")
    print(f"Duration:     {len(shifted) / audio_rate:.4f} s")
    print()
    print(
        "IMPORTANT: This is a diagnostic frequency-shifted "
        "motion signal, not recovered original audio."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Make low-frequency visual motion audible."
    )

    parser.add_argument(
        "video",
        help="Input video",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output WAV",
    )

    parser.add_argument(
        "--target",
        type=float,
        default=400.0,
        help="Audible carrier frequency in Hz (default: 400)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    if args.output is None:
        if args.video.lower().endswith(".mp4"):
            output = args.video[:-4] + "_motion_audible.wav"
        else:
            output = args.video + "_motion_audible.wav"
    else:
        output = args.output

    extract_and_shift(
        args.video,
        output,
        target_hz=args.target,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
