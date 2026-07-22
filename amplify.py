import os
import cv2
import numpy as np
from scipy.signal import butter
from scipy.ndimage import convolve, gaussian_filter, pyramid_reduce, pyramid_expand


class RieszVideoMagnifier:
    """
    Python translation of the Quaternionic Riesz Pyramid Pseudocode 
    by Neal Wadhwa et al. (MIT CSAIL, ICCP 2014).
    """

    def __init__(self, num_levels=3, alpha=15.0, low_cutoff=0.8, high_cutoff=2.5, fps=30.0):
        self.num_levels = num_levels
        self.alpha = alpha
        self.fps = fps
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

        # Riesz derivative kernels (Central Difference Approximations)
        self.kernel_x = np.array([[0.0, 0.0, 0.0],
                                  [0.5, 0.0, -0.5],
                                  [0.0, 0.0, 0.0]], dtype=np.float32)
        self.kernel_y = np.array([[0.0, 0.5, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, -0.5, 0.0]], dtype=np.float32)

        self._update_filter()
        self.reset_state()

    def _update_filter(self):
        """Updates 1st-order IIR Butterworth temporal bandpass filter."""
        nyquist = self.fps / 2.0
        low = np.clip(self.low_cutoff / nyquist, 0.001, 0.99)
        high = np.clip(self.high_cutoff / nyquist, low + 0.001, 0.999)
        self.b, self.a = butter(1, [low, high], btype='band')

    def set_parameters(self, alpha=None, low_cutoff=None, high_cutoff=None):
        """Dynamically update magnification and temporal filter settings."""
        if alpha is not None:
            self.alpha = alpha
        if low_cutoff is not None:
            self.low_cutoff = low_cutoff
        if high_cutoff is not None:
            self.high_cutoff = high_cutoff
        self._update_filter()

    def reset_state(self):
        """Clears frame history and phase registers."""
        self.initialized = False
        self.prev_lap = None
        self.prev_rx = None
        self.prev_ry = None

        self.phase_cos, self.phase_sin = [], []
        self.reg0_cos, self.reg1_cos = [], []
        self.reg0_sin, self.reg1_sin = [], []

    def build_laplacian_pyramid(self, image):
        """Builds Laplacian pyramid levels and lowpass residual."""
        pyramid = []
        current = image.astype(np.float32)

        for _ in range(self.num_levels):
            down = pyramid_reduce(current, channel_axis=None)
            up = pyramid_expand(down, channel_axis=None)

            if up.shape != current.shape:
                up = up[:current.shape[0], :current.shape[1]]

            lap = current - up
            pyramid.append(lap)
            current = down

        pyramid.append(current)  # Lowpass residual
        return pyramid

    def collapse_laplacian_pyramid(self, pyramid):
        """Reconstructs frame from Laplacian pyramid."""
        current = pyramid[-1]
        for lap in reversed(pyramid[:-1]):
            up = pyramid_expand(current, channel_axis=None)
            if up.shape != lap.shape:
                up = up[:lap.shape[0], :lap.shape[1]]
            current = lap + up
        return current

    def compute_riesz_pyramid(self, frame):
        """Computes Laplacian pyramid and 2D Riesz transform components (X, Y)."""
        lap_pyr = self.build_laplacian_pyramid(frame)
        riesz_x = [convolve(level, self.kernel_x, mode='nearest') for level in lap_pyr[:-1]]
        riesz_y = [convolve(level, self.kernel_y, mode='nearest') for level in lap_pyr[:-1]]
        return lap_pyr, riesz_x, riesz_y

    def iir_filter(self, phase, reg0, reg1):
        """Direct Form II IIR temporal filter."""
        filtered = self.b[0] * phase + reg0
        new_reg0 = self.b[1] * phase + reg1 - self.a[1] * filtered
        new_reg1 = self.b[2] * phase - self.a[2] * filtered
        return filtered, new_reg0, new_reg1

    def amplitude_weighted_blur(self, phase, amplitude, sigma=2.0):
        """Spatially smooths phase weighted by local amplitude."""
        num = gaussian_filter(phase * amplitude, sigma=sigma)
        den = gaussian_filter(amplitude, sigma=sigma) + 1e-8
        return num / den

    def compute_phase_diff(self, curr_r, curr_x, curr_y, prev_r, prev_x, prev_y):
        """Quaternion conjugate multiplication and logarithm for phase extraction."""
        q_real = curr_r * prev_r + curr_x * prev_x + curr_y * prev_y
        q_x = -curr_r * prev_x + prev_r * curr_x
        q_y = -curr_r * prev_y + prev_r * curr_y

        q_amp = np.sqrt(q_real**2 + q_x**2 + q_y**2) + 1e-8
        phase_diff = np.arccos(np.clip(q_real / q_amp, -1.0, 1.0))

        denom = np.sqrt(q_x**2 + q_y**2) + 1e-8
        cos_orient = q_x / denom
        sin_orient = q_y / denom

        phase_diff_cos = phase_diff * cos_orient
        phase_diff_sin = phase_diff * sin_orient
        amplitude = np.sqrt(q_amp)

        return phase_diff_cos, phase_diff_sin, amplitude

    def process_frame(self, frame_gray):
        """Processes a single grayscale frame [0.0, 1.0]."""
        curr_lap, curr_rx, curr_ry = self.compute_riesz_pyramid(frame_gray)

        if not self.initialized:
            self.prev_lap, self.prev_rx, self.prev_ry = curr_lap, curr_rx, curr_ry
            for k in range(self.num_levels):
                shape = curr_lap[k].shape
                self.phase_cos.append(np.zeros(shape, dtype=np.float32))
                self.phase_sin.append(np.zeros(shape, dtype=np.float32))
                self.reg0_cos.append(np.zeros(shape, dtype=np.float32))
                self.reg1_cos.append(np.zeros(shape, dtype=np.float32))
                self.reg0_sin.append(np.zeros(shape, dtype=np.float32))
                self.reg1_sin.append(np.zeros(shape, dtype=np.float32))
            self.initialized = True
            return frame_gray

        magnified_lap = []

        for k in range(self.num_levels):
            # 1. Quaternionic phase difference
            d_cos, d_sin, amp = self.compute_phase_diff(
                curr_lap[k], curr_rx[k], curr_ry[k],
                self.prev_lap[k], self.prev_rx[k], self.prev_ry[k]
            )

            # 2. Accumulate / unwrap phase
            self.phase_cos[k] += d_cos
            self.phase_sin[k] += d_sin

            # 3. IIR Temporal filter
            f_cos, self.reg0_cos[k], self.reg1_cos[k] = self.iir_filter(
                self.phase_cos[k], self.reg0_cos[k], self.reg1_cos[k]
            )
            f_sin, self.reg0_sin[k], self.reg1_sin[k] = self.iir_filter(
                self.phase_sin[k], self.reg0_sin[k], self.reg1_sin[k]
            )

            # 4. Amplitude-weighted spatial blur
            f_cos = self.amplitude_weighted_blur(f_cos, amp)
            f_sin = self.amplitude_weighted_blur(f_sin, amp)

            # 5. Phase amplification
            mag_cos = self.alpha * f_cos
            mag_sin = self.alpha * f_sin

            # 6. Reconstitute real part after phase shift
            phase_mag = np.sqrt(mag_cos**2 + mag_sin**2) + 1e-8
            exp_real = np.cos(phase_mag)
            exp_x = (mag_cos / phase_mag) * np.sin(phase_mag)
            exp_y = (mag_sin / phase_mag) * np.sin(phase_mag)

            res_real = exp_real * curr_lap[k] - exp_x * curr_rx[k] - exp_y * curr_ry[k]
            magnified_lap.append(res_real)

        # Append lowpass residual
        magnified_lap.append(curr_lap[-1])

        # Collapse pyramid
        out_frame = self.collapse_laplacian_pyramid(magnified_lap)
        out_frame = np.clip(out_frame, 0.0, 1.0)

        # Update previous frame
        self.prev_lap, self.prev_rx, self.prev_ry = curr_lap, curr_rx, curr_ry

        return out_frame


def run_motion_magnification(video_path, output_path="magnified_output.mp4"):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    # Initialize Riesz Magnifier
    magnifier = RieszVideoMagnifier(num_levels=3, alpha=15.0, low_cutoff=0.8, high_cutoff=2.5, fps=fps)

    # Interactive UI Controls
    window_name = "Riesz Motion Magnifier (Original vs Magnified)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Alpha (Amplification)", window_name, 15, 100, lambda x: None)
    cv2.createTrackbar("Low Cutoff (Hz x10)", window_name, 8, 100, lambda x: None)
    cv2.createTrackbar("High Cutoff (Hz x10)", window_name, 25, 100, lambda x: None)

    print("\n--- Starting Motion Magnification ---")
    print(f"Input: {video_path}")
    print("Press 'q' or 'ESC' to stop early.\n")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Read slider parameters
        alpha = cv2.getTrackbarPos("Alpha (Amplification)", window_name)
        low_c = max(1, cv2.getTrackbarPos("Low Cutoff (Hz x10)", window_name)) / 10.0
        high_c = max(low_c * 10 + 1, cv2.getTrackbarPos("High Cutoff (Hz x10)", window_name)) / 10.0

        magnifier.set_parameters(alpha=alpha, low_cutoff=low_c, high_cutoff=high_c)

        # Convert BGR -> YUV to amplify motion in Y (luminance channel)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0].astype(np.float32) / 255.0

        # Run Riesz Pyramid motion magnification
        y_magnified = magnifier.process_frame(y_channel)

        # Reconstruct BGR image
        yuv[:, :, 0] = np.clip(y_magnified * 255.0, 0, 255).astype(np.uint8)
        frame_out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # Save to video file
        out.write(frame_out)

        # Display side-by-side comparison
        combined_display = np.hstack((frame, frame_out))
        cv2.imshow(window_name, combined_display)

        frame_idx += 1
        if frame_idx % 15 == 0:
            print(f"Processing frame {frame_idx}/{total_frames if total_frames > 0 else 'Unknown'}")

        # Break loop on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Processing interrupted by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Magnified output saved as: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    # -------------------------------------------------------------
    # PASS YOUR VIDEO FILE PATH HERE
    # -------------------------------------------------------------
    input_video_file = "mute.mp4"  # <-- Change to your video filename
    output_video_file = "mute_magnified.mp4"

    run_motion_magnification(input_video_file, output_video_file)