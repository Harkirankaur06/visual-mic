import os
import numpy as np
from scipy.io.wavfile import write
from scipy.interpolate import interp1d

# ============================================================
# INPUT
# ============================================================

folder = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

input_file = os.path.join(
    folder,
    "GREEN_LINE_FRAME_DISPLACEMENT.npz"
)

# ============================================================
# LOAD GREEN-LINE DISPLACEMENT
# ============================================================

data = np.load(input_file)

# Use the displacement measured from frame 0
disp = data["y_displacement"].astype(np.float64)

fps_video = float(data["fps"])

print("Original frames:", len(disp))
print("Original video FPS:", fps_video)
print("Original duration:", len(disp) / fps_video)

# ============================================================
# FRAME 0 = ZERO
# ============================================================

disp = disp - disp[0]

# Remove only the overall DC offset.
# DO NOT differentiate.
# DO NOT filter.
# DO NOT add a carrier.
disp = disp - np.mean(disp)

# Normalize safely
peak = np.max(np.abs(disp))

if peak > 0:
    disp = disp / peak

# ============================================================
# SPEED-UP
# ============================================================

speed_factors = [5, 10, 20, 50]

sample_rate = 48000

for speed in speed_factors:

    # Original groove time
    t_original = np.arange(len(disp)) / fps_video

    # New playback duration
    new_duration = t_original[-1] / speed

    # Number of output samples
    n_samples = max(1000, int(new_duration * sample_rate))

    # New compressed time axis
    t_new = np.linspace(
        0,
        t_original[-1],
        n_samples
    )

    # Interpolate the ORIGINAL groove.
    interpolator = interp1d(
        t_original,
        disp,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    audio = interpolator(t_new)

    # Small fade to prevent clicks
    fade_len = min(
        int(0.005 * sample_rate),
        len(audio) // 2
    )

    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len)
        audio[:fade_len] *= fade
        audio[-fade_len:] *= fade[::-1]

    # Normalize
    peak_audio = np.max(np.abs(audio))

    if peak_audio > 0:
        audio = audio / peak_audio

    # Slight headroom
    audio = 0.9 * audio

    # Convert to 16-bit WAV
    wav = np.int16(audio * 32767)

    output_file = os.path.join(
        folder,
        f"GREEN_GROOVE_SPEED{speed}.wav"
    )

    write(
        output_file,
        sample_rate,
        wav
    )

    print()
    print("Speed:", speed, "x")
    print("Duration:", len(audio) / sample_rate, "seconds")
    print("Output:", output_file)

print()
print("DONE.")