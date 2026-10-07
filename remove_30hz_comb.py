import numpy as np
from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt


# ============================================================
# INPUT
# ============================================================

INPUT = Path(
    r"C:\Users\Harkiran\Downloads\speech_hybrid"
    r"\hybrid results\candidate_08_band_06.wav"
)

OUT_DIR = (
    INPUT.parent /
    "comb_removed"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

# Camera frame rate
FRAME_RATE = 30.000816

# Measured artifact residues:
#
# 96.316, 126.291, 156.267, ...
# 113.725, 143.701, 173.677, ...

RESIDUES = [
    6.3,
    23.7
]

# Width of each notch in Hz.
#
# The measured peaks are extremely narrow.
# Start conservatively.
NOTCH_HALF_WIDTH = 1.0

# Only remove comb components up to this frequency.
MAX_NOTCH_FREQ = 6000


# ============================================================
# LOAD
# ============================================================

fs, x = wavfile.read(
    str(INPUT)
)

x = x.astype(
    np.float64
)

if x.ndim > 1:
    x = np.mean(
        x,
        axis=1
    )

x -= np.mean(x)

peak = np.max(
    np.abs(x)
)

x /= (
    peak + 1e-12
)

print()
print("=" * 70)
print("30-HZ COMB SUPPRESSION")
print("=" * 70)
print(f"Input      : {INPUT}")
print(f"Sample rate: {fs}")
print(f"Samples    : {len(x)}")
print(f"Duration   : {len(x)/fs:.3f} s")
print()
print(
    f"Frame-rate artifact spacing: "
    f"{FRAME_RATE:.6f} Hz"
)
print(
    f"Notch half-width: "
    f"{NOTCH_HALF_WIDTH:.2f} Hz"
)
print()
print(
    "Artifact families:"
)
print(
    "  6.3 + 30n Hz"
)
print(
    "  23.7 + 30n Hz"
)
print("=" * 70)
print()


# ============================================================
# FFT
# ============================================================

N = len(x)

X = np.fft.rfft(x)

freq = np.fft.rfftfreq(
    N,
    d=1.0 / fs
)

original_power = (
    np.abs(X) ** 2
)


# ============================================================
# BUILD COMB MASK
# ============================================================

keep = np.ones(
    len(freq),
    dtype=bool
)

removed_frequencies = []

for residue in RESIDUES:

    n_min = int(
        np.floor(
            (20 - residue) /
            FRAME_RATE
        )
    )

    n_max = int(
        np.ceil(
            (MAX_NOTCH_FREQ - residue) /
            FRAME_RATE
        )
    )

    for n in range(
        max(0, n_min),
        n_max + 1
    ):

        center = (
            residue +
            n * FRAME_RATE
        )

        if center < 20:
            continue

        if center > MAX_NOTCH_FREQ:
            continue

        distance = np.abs(
            freq - center
        )

        mask = (
            distance <=
            NOTCH_HALF_WIDTH
        )

        if np.any(mask):

            keep[mask] = False

            removed_frequencies.append(
                center
            )


# ============================================================
# SUPPRESS COMB
# ============================================================

X_clean = X.copy()

X_clean[
    ~keep
] = 0


# ============================================================
# RECONSTRUCT
# ============================================================

clean = np.fft.irfft(
    X_clean,
    n=N
)

clean -= np.mean(
    clean
)

clean /= (
    np.max(
        np.abs(clean)
    ) + 1e-12
)


# ============================================================
# SAVE WAV
# ============================================================

output_wav = (
    OUT_DIR /
    "band08_30hz_comb_removed.wav"
)

pcm = np.int16(
    np.clip(
        clean * 0.9,
        -1,
        1
    ) * 32767
)

wavfile.write(
    str(output_wav),
    fs,
    pcm
)

print(
    f"Saved WAV:\n{output_wav}"
)

print()
print(
    f"Number of comb notches: "
    f"{len(removed_frequencies)}"
)


# ============================================================
# BEFORE / AFTER SPECTRUM
# ============================================================

# Limit display to 0-6000 Hz
display = (
    freq <= 6000
)

orig_db = (
    10 *
    np.log10(
        original_power +
        1e-14
    )
)

clean_power = (
    np.abs(X_clean) ** 2
)

clean_db = (
    10 *
    np.log10(
        clean_power +
        1e-14
    )
)


# ------------------------------------------------------------
# Original spectrum
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    freq[display],
    orig_db[display]
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Power (dB)"
)

plt.title(
    "Band 08 — Original Spectrum"
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

path = (
    OUT_DIR /
    "01_original_spectrum.png"
)

plt.savefig(
    path,
    dpi=180
)

plt.close()


# ------------------------------------------------------------
# Cleaned spectrum
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    freq[display],
    clean_db[display]
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Power (dB)"
)

plt.title(
    "Band 08 — After 30-Hz Comb Suppression"
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

path = (
    OUT_DIR /
    "02_comb_removed_spectrum.png"
)

plt.savefig(
    path,
    dpi=180
)

plt.close()


# ------------------------------------------------------------
# Difference
# ------------------------------------------------------------

difference = (
    orig_db -
    clean_db
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    freq[display],
    difference[display]
)

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Power removed (dB)"
)

plt.title(
    "Spectrum Difference — Removed 30-Hz Comb"
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

path = (
    OUT_DIR /
    "03_removed_comb.png"
)

plt.savefig(
    path,
    dpi=180
)

plt.close()


# ============================================================
# SPECTROGRAM BEFORE
# ============================================================

plt.figure(
    figsize=(13, 5)
)

plt.specgram(
    x,
    NFFT=4096,
    Fs=fs,
    noverlap=3072
)

plt.ylim(
    0,
    5000
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Frequency (Hz)"
)

plt.title(
    "Band 08 — Before Comb Suppression"
)

plt.colorbar(
    label="Power"
)

plt.tight_layout()

path = (
    OUT_DIR /
    "04_before_spectrogram.png"
)

plt.savefig(
    path,
    dpi=180
)

plt.close()


# ============================================================
# SPECTROGRAM AFTER
# ============================================================

plt.figure(
    figsize=(13, 5)
)

plt.specgram(
    clean,
    NFFT=4096,
    Fs=fs,
    noverlap=3072
)

plt.ylim(
    0,
    5000
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Frequency (Hz)"
)

plt.title(
    "Band 08 — After 30-Hz Comb Suppression"
)

plt.colorbar(
    label="Power"
)

plt.tight_layout()

path = (
    OUT_DIR /
    "05_after_spectrogram.png"
)

plt.savefig(
    path,
    dpi=180
)

plt.close()


# ============================================================
# RMS
# ============================================================

original_rms = np.sqrt(
    np.mean(
        x ** 2
    )
)

clean_rms = np.sqrt(
    np.mean(
        clean ** 2
    )
)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print(
    f"Original RMS : "
    f"{original_rms:.6f}"
)

print(
    f"Clean RMS    : "
    f"{clean_rms:.6f}"
)

print()
print(
    f"Results folder:"
)
print(
    OUT_DIR
)

print()
print(
    "Listen to:"
)
print(
    "  band08_30hz_comb_removed.wav"
)

print()
print(
    "Inspect:"
)
print(
    "  02_comb_removed_spectrum.png"
)
print(
    "  05_after_spectrogram.png"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)