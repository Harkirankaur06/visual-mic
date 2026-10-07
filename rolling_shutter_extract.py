import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.io import wavfile
import os

# ============================================================
# ROLLING-SHUTTER AUDIO EXTRACTION — A15 440 Hz TEST
# ============================================================

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\A15_440Hz_test_clean.mp4"

# From your calibration experiment
ROW_DELAY = 8.3763e-6       # seconds / row

# Expected test tone
TARGET_FREQ = 440.0         # Hz

# Output audio sampling rate
OUTPUT_FS = 48000

# Number of horizontal row bands
NUM_BANDS = 240

# Ignore extreme top/bottom regions
TOP_MARGIN = 0.10
BOTTOM_MARGIN = 0.90

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video:\n{VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = frame_count / fps

print("\n==============================================")
print("ROLLING-SHUTTER AUDIO EXTRACTION")
print("==============================================")
print(f"Video       : {VIDEO}")
print(f"Resolution  : {width} x {height}")
print(f"FPS         : {fps:.6f}")
print(f"Frames      : {frame_count}")
print(f"Duration    : {duration:.4f} s")
print(f"Row delay   : {ROW_DELAY * 1e6:.4f} us")
print(f"Target tone : {TARGET_FREQ:.1f} Hz")

# ============================================================
# SELECT ROW REGION
# ============================================================

y_start = int(height * TOP_MARGIN)
y_end = int(height * BOTTOM_MARGIN)

rows = np.linspace(
    y_start,
    y_end - 1,
    NUM_BANDS
).astype(int)

print(f"Rows used   : {y_start} - {y_end}")
print(f"Row bands   : {NUM_BANDS}")

# ============================================================
# EXTRACT ROW-BAND SIGNALS
# ============================================================

signals = []

frame_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize each frame to reduce lighting drift
    gray = gray.astype(np.float64)

    # Central 60% of image
    x1 = int(width * 0.20)
    x2 = int(width * 0.80)

    frame_signal = []

    for y in rows:

        # Small vertical neighborhood around row
        y1 = max(0, y - 2)
        y2 = min(height, y + 3)

        value = np.mean(gray[y1:y2, x1:x2])

        frame_signal.append(value)

    signals.append(frame_signal)

    frame_index += 1

    if frame_index % 25 == 0:
        print(f"Processed {frame_index}/{frame_count} frames")

cap.release()

signals = np.asarray(signals)

print("\nRaw matrix:")
print("Frames x RowBands =", signals.shape)

# ============================================================
# REMOVE DC / SLOW LIGHTING VARIATION
# ============================================================

signals -= np.mean(signals, axis=0, keepdims=True)

# Remove slow temporal drift from each row band
if len(signals) > 20:

    # Remove very slow lighting drift.
    # The cutoff must remain below the frame-rate Nyquist.
    slow_cutoff = min(5.0, fps * 0.25)

    slow_sos = butter(
        2,
        slow_cutoff,
        btype="highpass",
        fs=fps,
        output="sos"
    )

    frame_filtered = sosfiltfilt(
        slow_sos,
        signals,
        axis=0
    )

    # Only use this for diagnostic frame-rate signal.
    # The actual rolling-shutter reconstruction below
    # uses row timing.
    frame_filtered = sosfiltfilt(
        slow_sos,
        signals,
        axis=0
    )

else:
    frame_filtered = signals

# ============================================================
# ROLLING-SHUTTER TIME RECONSTRUCTION
# ============================================================

#
# Each row is captured at a slightly different time.
#
# t(frame,row) =
#
#     frame_index / FPS
#     +
#     row_position * ROW_DELAY
#
# ============================================================

timestamps = []
values = []

for f in range(frame_filtered.shape[0]):

    frame_start_time = f / fps

    for r in range(NUM_BANDS):

        row_position = rows[r] - y_start

        t = (
            frame_start_time
            + row_position * ROW_DELAY
        )

        timestamps.append(t)
        values.append(frame_filtered[f, r])

timestamps = np.asarray(timestamps)
values = np.asarray(values)

# ============================================================
# SORT BY TIME
# ============================================================

order = np.argsort(timestamps)

timestamps = timestamps[order]
values = values[order]

# ============================================================
# REMOVE DUPLICATE / INVALID TIMESTAMPS
# ============================================================

valid = np.isfinite(timestamps) & np.isfinite(values)

timestamps = timestamps[valid]
values = values[valid]

# ============================================================
# INTERPOLATE TO UNIFORM AUDIO GRID
# ============================================================

t_start = timestamps[0]
t_end = timestamps[-1]

uniform_t = np.arange(
    t_start,
    t_end,
    1.0 / OUTPUT_FS
)

audio = np.interp(
    uniform_t,
    timestamps,
    values
)

# ============================================================
# NORMALIZE
# ============================================================

audio -= np.mean(audio)

peak = np.max(np.abs(audio))

if peak > 0:
    audio /= peak

# ============================================================
# BANDPASS AROUND 440 Hz
# ============================================================

low = 300.0
high = 600.0

sos = butter(
    4,
    [low, high],
    btype="bandpass",
    fs=OUTPUT_FS,
    output="sos"
)

audio_440 = sosfiltfilt(sos, audio)

peak = np.max(np.abs(audio_440))

if peak > 0:
    audio_440 /= peak

# ============================================================
# SAVE WAV
# ============================================================

output_dir = os.path.dirname(VIDEO)

full_wav = os.path.join(
    output_dir,
    "A15_440Hz_rolling_recovered_full.wav"
)

tone_wav = os.path.join(
    output_dir,
    "A15_440Hz_rolling_recovered_300_600Hz.wav"
)

wavfile.write(
    full_wav,
    OUTPUT_FS,
    np.int16(
        np.clip(audio, -1, 1) * 32767
    )
)

wavfile.write(
    tone_wav,
    OUTPUT_FS,
    np.int16(
        np.clip(audio_440, -1, 1) * 32767
    )
)

# ============================================================
# FFT / SPECTRAL ANALYSIS
# ============================================================

analysis_signal = audio

if len(analysis_signal) > OUTPUT_FS:

    nperseg = min(
        65536,
        len(analysis_signal)
    )

    freqs, power = welch(
        analysis_signal,
        fs=OUTPUT_FS,
        nperseg=nperseg
    )

    # Look around expected 440 Hz
    search = (
        (freqs >= 300) &
        (freqs <= 600)
    )

    if np.any(search):

        local_freqs = freqs[search]
        local_power = power[search]

        peak_index = np.argmax(local_power)

        detected_freq = local_freqs[peak_index]

        target_index = np.argmin(
            np.abs(freqs - TARGET_FREQ)
        )

        target_power = power[target_index]

        noise_region = (
            (freqs >= 300) &
            (freqs <= 600) &
            (np.abs(freqs - TARGET_FREQ) > 20)
        )

        if np.any(noise_region):

            noise_floor = np.median(
                power[noise_region]
            )

            if noise_floor > 0:
                snr_db = 10 * np.log10(
                    target_power / noise_floor
                )
            else:
                snr_db = float("inf")
        else:
            snr_db = float("nan")

        print("\n==============================================")
        print("SPECTRAL RESULT")
        print("==============================================")
        print(f"Expected frequency : {TARGET_FREQ:.2f} Hz")
        print(f"Detected peak      : {detected_freq:.2f} Hz")
        print(f"Approx. SNR        : {snr_db:.2f} dB")

        if abs(detected_freq - TARGET_FREQ) <= 10:

            print("\n>>> 440 Hz PEAK DETECTED <<<")
            print("The rolling-shutter reconstruction")
            print("contains a spectral component near")
            print("the known test tone.")

        else:

            print("\n>>> NO CLEAR 440 Hz PEAK <<<")
            print("The rolling-shutter reconstruction")
            print("does not yet show the expected tone.")

# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n==============================================")
print("OUTPUT")
print("==============================================")

print(f"Full reconstructed signal:")
print(full_wav)

print("\n300–600 Hz filtered signal:")
print(tone_wav)

print("\nDone.")