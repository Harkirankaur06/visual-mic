import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os

# ============================================================
# INPUT
# ============================================================

DATA = r"C:\Users\Harkiran\Downloads\newlightvibeexp\GREEN_LINE_FRAME_DISPLACEMENT.npz"

OUTDIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

FS = 48000

# ============================================================
# LOAD GREEN-LINE GROOVE
# ============================================================

d = np.load(DATA)

frames = d["frame"]
displacement = d["y_displacement"].astype(float)
fps = float(d["fps"])

print("Frames:", len(frames))
print("FPS:", fps)

# Frame 0 = zero
displacement -= displacement[0]

# ============================================================
# PLOT THE GROOVE
# ============================================================

plt.figure(figsize=(15, 6))

plt.plot(
    frames,
    displacement,
    linewidth=2
)

plt.axhline(0, linewidth=1)

plt.xlabel("Frame")
plt.ylabel("Displacement from frame 0 (pixels)")

plt.title("GREEN LINE AS PHYSICAL GROOVE")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "GREEN_LINE_GROOVE.png"
    ),
    dpi=200
)

plt.close()

# ============================================================
# NORMALIZE
# ============================================================

groove = displacement.copy()

groove -= np.mean(groove)

scale = np.max(np.abs(groove))

if scale > 0:
    groove /= scale

# ============================================================
# RESAMPLE THE GROOVE
#
# This does NOT change the shape.
# It only gives us enough samples to write an audio file.
# ============================================================

duration = len(groove) / fps

audio_time = np.linspace(
    0,
    duration,
    int(duration * FS),
    endpoint=False
)

frame_time = np.arange(len(groove)) / fps

g = np.interp(
    audio_time,
    frame_time,
    groove
)

# ============================================================
# METHOD 1
# RAW GROOVE -> FREQUENCY
#
# Larger displacement = higher/lower instantaneous pitch.
# The actual green-line movement controls the pitch.
# ============================================================

MIN_FREQ = 100.0
MAX_FREQ = 1200.0

frequency = (
    MIN_FREQ
    +
    (g + 1.0) / 2.0
    *
    (MAX_FREQ - MIN_FREQ)
)

phase = np.cumsum(
    2.0 * np.pi * frequency / FS
)

audio_groove = np.sin(phase)

# ============================================================
# METHOD 2
# VELOCITY GROOVE
#
# The gramophone analogy:
# movement between successive positions becomes the signal.
# ============================================================

velocity = np.gradient(g)

velocity -= np.mean(velocity)

vscale = np.max(np.abs(velocity))

if vscale > 0:
    velocity /= vscale

frequency_v = (
    MIN_FREQ
    +
    (velocity + 1.0) / 2.0
    *
    (MAX_FREQ - MIN_FREQ)
)

phase_v = np.cumsum(
    2.0 * np.pi * frequency_v / FS
)

audio_velocity = np.sin(phase_v)

# ============================================================
# METHOD 3
# ACCELERATION GROOVE
# ============================================================

acceleration = np.gradient(velocity)

acceleration -= np.mean(acceleration)

ascale = np.max(np.abs(acceleration))

if ascale > 0:
    acceleration /= ascale

frequency_a = (
    MIN_FREQ
    +
    (acceleration + 1.0) / 2.0
    *
    (MAX_FREQ - MIN_FREQ)
)

phase_a = np.cumsum(
    2.0 * np.pi * frequency_a / FS
)

audio_acceleration = np.sin(phase_a)

# ============================================================
# NORMALIZE + SAVE
# ============================================================

def save_audio(path, audio):

    audio = audio.astype(float)

    audio -= np.mean(audio)

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio /= peak

    audio *= 0.95

    write(
        path,
        FS,
        np.int16(audio * 32767)
    )


save_audio(
    os.path.join(
        OUTDIR,
        "GREEN_GROOVE_RAW.wav"
    ),
    audio_groove
)

save_audio(
    os.path.join(
        OUTDIR,
        "GREEN_GROOVE_VELOCITY.wav"
    ),
    audio_velocity
)

save_audio(
    os.path.join(
        OUTDIR,
        "GREEN_GROOVE_ACCELERATION.wav"
    ),
    audio_acceleration
)

# ============================================================
# PLOT THE THREE GROOVE SIGNALS
# ============================================================

plt.figure(figsize=(15, 6))

plt.plot(
    audio_time,
    g,
    linewidth=1
)

plt.xlabel("Time (seconds)")
plt.ylabel("Normalized displacement")

plt.title("Green-Line Groove")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTDIR,
        "GREEN_GROOVE_SIGNAL.png"
    ),
    dpi=200
)

plt.close()

print()
print("CREATED:")
print("GREEN_LINE_GROOVE.png")
print("GREEN_GROOVE_SIGNAL.png")
print("GREEN_GROOVE_RAW.wav")
print("GREEN_GROOVE_VELOCITY.wav")
print("GREEN_GROOVE_ACCELERATION.wav")
print()
print("DONE")