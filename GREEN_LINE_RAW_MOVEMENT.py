import numpy as np
from scipy.io.wavfile import write

DATA = r"C:\Users\Harkiran\Downloads\newlightvibeexp\GREEN_LINE_coordinates.npz"
OUT = r"C:\Users\Harkiran\Downloads\newlightvibeexp\GREEN_LINE_RAW_MOVEMENT.wav"

d = np.load(DATA)

movement = d["movement"].astype(float)
fps = float(d["fps"])

# This is ONLY the measured Y movement.
# No carrier.
# No filtering.
# No frequency shifting.

movement -= np.mean(movement)

# Normalize
peak = np.max(np.abs(movement))

if peak > 0:
    movement /= peak

# Save at the ACTUAL measurement rate.
# This preserves the measured waveform exactly.
write(
    OUT,
    int(round(fps)),
    np.int16(movement * 32767)
)

print("Saved:", OUT)
print("Sampling rate:", fps)
print("Samples:", len(movement))
print("Duration:", len(movement) / fps)
print("Y movement range:", movement.min(), movement.max())