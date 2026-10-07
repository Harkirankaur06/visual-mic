import cv2
import numpy as np
import sys
import os


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps > 0 else 0

    print("=" * 60)
    print("ROLLING-SHUTTER VIDEO PROBE")
    print("=" * 60)
    print(f"Video       : {video_path}")
    print(f"Resolution  : {width} x {height}")
    print(f"FPS         : {fps:.6f}")
    print(f"Frames      : {frame_count}")
    print(f"Duration    : {duration:.6f} s")
    print(f"Frame period: {1000.0 / fps:.6f} ms")
    print("=" * 60)

    # ---------------------------------------------------------
    # Read a small number of frames
    # ---------------------------------------------------------

    frames = []

    max_frames = min(frame_count, 30)

    for i in range(max_frames):
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize to float
        gray = gray.astype(np.float32) / 255.0

        frames.append(gray)

    cap.release()

    if len(frames) < 2:
        raise RuntimeError("Not enough frames for analysis.")

    frames = np.asarray(frames)

    print(f"Frames analyzed: {len(frames)}")

    # ---------------------------------------------------------
    # 1. Frame-to-frame motion
    # ---------------------------------------------------------

    temporal_difference = np.abs(frames[1:] - frames[:-1])

    global_motion = np.mean(temporal_difference, axis=(1, 2))

    print()
    print("FRAME-TO-FRAME MOTION")
    print("-" * 60)

    print(f"Mean motion : {np.mean(global_motion):.8f}")
    print(f"Std motion  : {np.std(global_motion):.8f}")
    print(f"Max motion  : {np.max(global_motion):.8f}")

    # ---------------------------------------------------------
    # 2. Row-wise variation
    # ---------------------------------------------------------

    # Average temporal difference for every image row.
    row_motion = np.mean(temporal_difference, axis=(0, 2))

    print()
    print("ROW-WISE MOTION")
    print("-" * 60)

    print(f"Number of rows : {height}")
    print(f"Mean row motion: {np.mean(row_motion):.8f}")
    print(f"Std row motion : {np.std(row_motion):.8f}")
    print(f"Max row motion : {np.max(row_motion):.8f}")

    # ---------------------------------------------------------
    # 3. Divide image into row blocks
    # ---------------------------------------------------------

    num_blocks = 20
    block_size = height // num_blocks

    print()
    print("ROW BLOCK ANALYSIS")
    print("-" * 60)

    block_values = []

    for b in range(num_blocks):

        start = b * block_size

        if b == num_blocks - 1:
            end = height
        else:
            end = (b + 1) * block_size

        value = np.mean(row_motion[start:end])

        block_values.append(value)

        print(
            f"Rows {start:4d}-{end-1:4d} : "
            f"{value:.8f}"
        )

    # ---------------------------------------------------------
    # 4. Compare top vs bottom of frame
    # ---------------------------------------------------------

    top = np.mean(row_motion[:height // 4])
    middle = np.mean(
        row_motion[height // 4:3 * height // 4]
    )
    bottom = np.mean(row_motion[3 * height // 4:])

    print()
    print("VERTICAL MOTION DISTRIBUTION")
    print("-" * 60)

    print(f"Top quarter    : {top:.8f}")
    print(f"Middle half    : {middle:.8f}")
    print(f"Bottom quarter : {bottom:.8f}")

    # ---------------------------------------------------------
    # 5. Estimate theoretical row sampling rates
    # ---------------------------------------------------------

    print()
    print("THEORETICAL ROW TIMING")
    print("-" * 60)

    frame_period = 1.0 / fps

    print(
        "IMPORTANT: These are NOT measured camera parameters."
    )
    print(
        "They are hypothetical values for understanding "
        "possible rolling-shutter sampling."
    )

    for row_delay_us in [5, 10, 20, 30, 50, 100]:

        row_delay = row_delay_us * 1e-6

        row_rate = 1.0 / row_delay
        nyquist = row_rate / 2.0

        print(
            f"{row_delay_us:3d} us/row -> "
            f"{row_rate:8.1f} rows/s -> "
            f"Nyquist {nyquist:8.1f} Hz"
        )

    # ---------------------------------------------------------
    # 6. Save row-motion profile
    # ---------------------------------------------------------

    output_dir = os.path.dirname(os.path.abspath(video_path))

    output_file = os.path.join(
        output_dir,
        "rolling_shutter_row_motion.npy"
    )

    np.save(output_file, row_motion)

    print()
    print("=" * 60)
    print(f"Saved row-motion profile:")
    print(output_file)
    print("=" * 60)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python rolling_shutter_probe.py "
            "\"video.mp4\""
        )
        sys.exit(1)

    analyze_video(sys.argv[1])