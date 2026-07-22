import os
import cv2
import numpy as np
import pyrtools as pt
import scipy.signal as signal
import soundfile as sf


def build_complex_steerable_pyramid(frame_gray, num_scales=4, num_orientations=4):
    """
    Builds a complex steerable pyramid on a grayscale image frame
    equivalent to buildSCFpyr from matlabPyrTools in the MIT release.
    """
    # Create steerable pyramid object using pyrtools
    pyr = pt.pyramids.SteerablePyramidFreq(
        frame_gray,
        height=num_scales,
        order=num_orientations - 1,
        is_complex=True
    )
    return pyr.pyr_coeffs


def extract_raw_audio_signals(video_path, num_scales=3, num_orientations=4):
    """
    Reads video frames and extracts local temporal phase variations across 
    spatial scales and orientations (translating vmSoundFromVideo.m).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 2200.0  # Default fallback high-speed frame rate if not set
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Reading video: {video_path}")
    print(f"FPS: {fps} | Total Frames: {total_frames}")

    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame from the video.")

    # Convert first frame to grayscale float32
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Build pyramid on reference frame to get subband dimensions and keys
    ref_coeffs = build_complex_steerable_pyramid(first_gray, num_scales, num_orientations)
    
    # Filter band keys to only get highpass complex directional bands
    band_keys = [k for k in ref_coeffs.keys() if isinstance(k, tuple)]

    # Pre-allocate phase signal storage
    num_bands = len(band_keys)
    phase_signals = {k: np.zeros((total_frames, *ref_coeffs[k].shape), dtype=np.float32) for k in band_keys}

    # Extract phase of reference frame
    for k in band_keys:
        phase_signals[k][0] = np.angle(ref_coeffs[k])

    print("Extracting phase temporal signals across video frames...")
    frame_idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        coeffs = build_complex_steerable_pyramid(gray, num_scales, num_orientations)

        # Store phase variation relative to initial reference frame
        for k in band_keys:
            phase_signals[k][frame_idx] = np.angle(coeffs[k])

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"Processed frame {frame_idx}/{total_frames}")

    cap.release()
    return phase_signals, fps, ref_coeffs


def align_and_combine_signals(phase_signals, ref_coeffs):
    """
    Weights and aligns multi-scale phase signals based on local amplitude/energy
    (translating vmAlignAToB.m and subband weighted averaging).
    """
    combined_signal = None

    print("Aligning and combining multi-band phase signals...")
    for k, signal_tensor in phase_signals.items():
        # Compute spatial weight based on magnitude energy of coefficients
        amplitude_weight = np.abs(ref_coeffs[k])
        
        # Flatten spatial pixels into weighted 1D temporal signal
        # shape: (num_frames, H, W) -> (num_frames,)
        weighted_1d = np.sum(signal_tensor * amplitude_weight, axis=(1, 2)) / (np.sum(amplitude_weight) + 1e-8)

        # Zero-mean alignment (unwrap local phase differences)
        unwrapped = np.unwrap(weighted_1d)
        unwrapped_diff = np.diff(unwrapped, prepend=unwrapped[0])

        if combined_signal is None:
            combined_signal = unwrapped_diff
        else:
            # Align signals (phase inversion check via cross-correlation)
            correlation = np.correlate(combined_signal, unwrapped_diff, mode='valid')[0]
            if correlation < 0:
                unwrapped_diff = -unwrapped_diff
            combined_signal += unwrapped_diff

    return combined_signal


def process_and_clean_audio(raw_audio, fps, low_cutoff=80.0, high_cutoff=1000.0, output_sample_rate=16000):
    """
    Applies Butterworth temporal bandpass filtering and resamples the extracted 
    audio to a standard sound rate (translating vmGetSoundSpecSub.m & STFT resynthesis).
    """
    print("Post-processing extracted sound signal...")
    
    # 1. High-pass / Bandpass filtering
    nyquist = fps / 2.0
    low = np.clip(low_cutoff / nyquist, 0.001, 0.99)
    high = np.clip(high_cutoff / nyquist, low + 0.001, 0.999)
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, raw_audio)

    # 2. Resample signal to standard 16 kHz audio rate
    num_output_samples = int(len(filtered_audio) * (output_sample_rate / fps))
    resampled_audio = signal.resample(filtered_audio, num_output_samples)

    # 3. Peak normalization to [-1.0, 1.0] range
    max_val = np.max(np.abs(resampled_audio))
    if max_val > 0:
        normalized_audio = resampled_audio / max_val
    else:
        normalized_audio = resampled_audio

    return normalized_audio


def visual_microphone_audio_extraction(video_path, output_audio_path="RecoveredSound_python.wav"):
    """Main execution function to extract sound from video."""
    # 1. Extract raw phase signals across pyramid scales
    phase_signals, fps, ref_coeffs = extract_raw_audio_signals(video_path)

    # 2. Combine multi-scale signals
    raw_sound = align_and_combine_signals(phase_signals, ref_coeffs)

    # 3. Filter and normalize audio waveform
    clean_sound = process_and_clean_audio(raw_sound, fps=fps)

    # 4. Save to WAV audio file
    sf.write(output_audio_path, clean_sound, 16000)
    print(f"\nSuccess! Recovered audio saved to: {os.path.abspath(output_audio_path)}")


# --- Execution Example ---
if __name__ == "__main__":
    # Specify your amplified or high-speed video path here
    input_video = "mute_magnified.mp4"  # <-- Pass your input video filename here
    output_wav = "RecoveredSound_python.wav"

    visual_microphone_audio_extraction(input_video, output_wav)