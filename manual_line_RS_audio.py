import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.io.wavfile import write
import os

VIDEO = r"C:\Users\Harkiran\Downloads\newlightvibeexp\MANUAL_LINE_tracking.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

ROW_DELAY = 7.7435e-6
AUDIO_FS = 48000

OUT_WAV = os.path.join(OUTDIR, "MANUAL_LINE_RS_AUDIO.wav")
OUT_WAVE = os.path.join(OUTDIR, "MANUAL_LINE_RS_WAVEFORM.png")
OUT_SPEC = os.path.join(OUTDIR, "MANUAL_LINE_RS_SPECTRUM.png")
OUT_PROFILE = os.path.join(OUTDIR, "MANUAL_LINE_RS_PROFILES.png")

# ------------------------------------------------------------
# The marked line is here.
# This is ONLY used to locate the physical reflection.
# ------------------------------------------------------------

X1 = 700
X2 = 1050

Y1 = 750
Y2 = 900

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Frames:", nframes)

profiles = []
centers = []

for frame_no in range(nframes):

    ret, frame = cap.read()

    if not ret:
        break

    roi = frame[Y1:Y2, X1:X2]

    b = roi[:, :, 0].astype(np.float32)
    g = roi[:, :, 1].astype(np.float32)
    r = roi[:, :, 2].astype(np.float32)

    # --------------------------------------------------------
    # FIND GREEN TRACKING MARKER
    # --------------------------------------------------------

    green = (
        (g > 100) &
        (g > r + 40) &
        (g > b + 40)
    )

    ys, xs = np.where(green)

    if len(ys) < 10:
        centers.append(np.nan)
        profiles.append(np.full(Y2-Y1, np.nan))
        continue

    center = np.median(ys)

    centers.append(center + Y1)

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove GREEN overlay from measurement.
    #
    # We now measure the ORIGINAL physical image underneath.
    # --------------------------------------------------------

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gray[green] = np.nan

    # Average brightness horizontally.
    # This gives a vertical brightness profile.
    profile = np.nanmedian(gray, axis=1)

    profiles.append(profile)

cap.release()

profiles = np.asarray(profiles, dtype=np.float64)
centers = np.asarray(centers, dtype=np.float64)

# ------------------------------------------------------------
# FILL MISSING CENTERS
# ------------------------------------------------------------

valid = np.isfinite(centers)

if valid.sum() < 20:
    raise RuntimeError("Could not locate the green tracking line.")

centers = np.interp(
    np.arange(len(centers)),
    np.where(valid)[0],
    centers[valid]
)

# ------------------------------------------------------------
# REMOVE INVALID PROFILE VALUES
# ------------------------------------------------------------

for i in range(len(profiles)):

    p = profiles[i]

    good = np.isfinite(p)

    if good.sum() < 10:
        profiles[i] = 0
        continue

    profiles[i] = np.interp(
        np.arange(len(p)),
        np.where(good)[0],
        p[good]
    )

print("Profile matrix:", profiles.shape)

# ------------------------------------------------------------
# SAVE PROFILE IMAGE
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

plt.imshow(
    profiles.T,
    aspect="auto",
    origin="upper",
    extent=[
        0,
        len(profiles) / fps,
        Y1,
        Y2
    ]
)

plt.xlabel("Time (seconds)")
plt.ylabel("Image Y")
plt.title("Physical Brightness Profile Around Manually Tracked Line")

plt.colorbar(label="Brightness")

plt.tight_layout()
plt.savefig(OUT_PROFILE, dpi=180)
plt.close()

# ------------------------------------------------------------
# REMOVE STATIC PROFILE
#
# This is NOT an audio filter.
# We only remove the part that never changes.
# ------------------------------------------------------------

reference = np.median(profiles, axis=0)

residual = profiles - reference[None, :]

# ------------------------------------------------------------
# EDGE / VIBRATION SIGNAL
#
# Vertical gradient tells us where the physical
# reflected-light boundary is.
# ------------------------------------------------------------

gradient = np.gradient(profiles, axis=1)

# Weight residual by physical edge strength
edge_strength = np.median(
    np.abs(gradient),
    axis=0
)

# Keep strongest physical edges
threshold = np.percentile(edge_strength, 70)

weights = np.maximum(
    edge_strength - threshold,
    0
)

if np.sum(weights) == 0:
    weights = edge_strength

weights = weights / (np.sum(weights) + 1e-12)

# ------------------------------------------------------------
# CONSTRUCT ROW-TIME SAMPLES
#
# Every image row has a slightly different exposure time.
# ------------------------------------------------------------

times = []
values = []

for frame_no in range(len(residual)):

    frame_start = frame_no / fps

    for row in range(residual.shape[1]):

        value = residual[frame_no, row]

        if not np.isfinite(value):
            continue

        global_row = Y1 + row

        t = frame_start + global_row * ROW_DELAY

        # Give stronger physical edges more influence
        v = value * weights[row]

        times.append(t)
        values.append(v)

times = np.asarray(times)
values = np.asarray(values)

order = np.argsort(times)

times = times[order]
values = values[order]

print("Rolling shutter samples:", len(values))
print("Time:", times.min(), "to", times.max())

# ------------------------------------------------------------
# REMOVE MEAN ONLY
# ------------------------------------------------------------

values -= np.mean(values)

# ------------------------------------------------------------
# INTERPOLATE TO AUDIO RATE
# ------------------------------------------------------------

audio_time = np.arange(
    times[0],
    times[-1],
    1.0 / AUDIO_FS
)

audio = np.interp(
    audio_time,
    times,
    values
)

audio -= np.mean(audio)

# ------------------------------------------------------------
# NORMALIZE
# ------------------------------------------------------------

peak = np.max(np.abs(audio))

if peak > 1e-12:
    audio = audio / peak

audio *= 0.95

# ------------------------------------------------------------
# WRITE WAV
# ------------------------------------------------------------

write(
    OUT_WAV,
    AUDIO_FS,
    np.int16(
        np.clip(audio, -1, 1) * 32767
    )
)

print()
print("WAV:")
print(OUT_WAV)

print("Duration:", len(audio) / AUDIO_FS)
print("RMS:", np.sqrt(np.mean(audio ** 2)))

# ------------------------------------------------------------
# WAVEFORM
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

plt.plot(
    audio_time,
    audio,
    linewidth=0.5
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.title(
    "Rolling-Shutter Waveform From Physical Vibration"
)

plt.grid(True)

plt.tight_layout()
plt.savefig(OUT_WAVE, dpi=180)
plt.close()

# ------------------------------------------------------------
# SPECTRUM
# ------------------------------------------------------------

nperseg = min(131072, len(audio))

f, p = welch(
    audio,
    fs=AUDIO_FS,
    nperseg=nperseg
)

plt.figure(figsize=(15, 5))

plt.plot(f, p)

plt.xlim(0, 3000)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.title(
    "Rolling-Shutter Spectrum From Physical Vibration"
)

plt.grid(True)

plt.tight_layout()
plt.savefig(OUT_SPEC, dpi=180)
plt.close()

# ------------------------------------------------------------
# TOP FREQUENCIES
# ------------------------------------------------------------

mask = (f >= 20) & (f <= 3000)

inds = np.argsort(p[mask])[::-1]

freqs = f[mask]
powers = p[mask]

print()
print("TOP PHYSICAL-SIGNAL FREQUENCIES:")

for i in inds[:25]:

    print(
        f"{freqs[i]:9.2f} Hz   "
        f"{powers[i]:.8e}"
    )

print()
print("DONE")