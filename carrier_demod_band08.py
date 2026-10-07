import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT = Path(
    r"C:\Users\Harkiran\OneDrive\visual mic"
    r"\\speech_hybrid\\candidate_08_band_06.wav"
)

OUT_DIR = INPUT.parent

# ============================================================
# LOAD
# ============================================================

fs, x = wavfile.read(INPUT)

x = x.astype(np.float64)

if x.ndim > 1:
    x = np.mean(x, axis=1)

x -= np.mean(x)

x /= np.max(np.abs(x)) + 1e-12

print()
print("============================================================")
print("BAND 08 CARRIER ANALYSIS")
print("============================================================")
print(f"Input     : {INPUT}")
print(f"Sample Hz : {fs}")
print(f"Samples   : {len(x)}")
print(f"Duration  : {len(x)/fs:.3f} s")
print("============================================================")
print()

# ============================================================
# 1. FIND DOMINANT FREQUENCIES
# ============================================================

nperseg = min(262144, len(x))

f, pxx = signal.welch(
    x,
    fs=fs,
    nperseg=nperseg,
    noverlap=nperseg // 2
)

# Ignore very low frequencies
valid = f >= 50

# Find peaks
peaks, properties = signal.find_peaks(
    pxx[valid],
    prominence=np.max(pxx[valid]) * 0.001
)

peak_indices = np.where(valid)[0][peaks]

# Sort strongest first
peak_indices = peak_indices[
    np.argsort(
        pxx[peak_indices]
    )[::-1]
]

print("Strongest spectral components:")

for i, idx in enumerate(peak_indices[:20]):

    print(
        f"{i+1:02d}. "
        f"{f[idx]:9.2f} Hz   "
        f"power={pxx[idx]:.4e}"
    )

# ============================================================
# 2. SAVE SPECTRUM
# ============================================================

plt.figure(figsize=(12, 5))

plt.semilogy(
    f,
    pxx + 1e-15
)

plt.xlim(0, min(8000, fs / 2))

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title(
    "Band 08 Spectrum - Carrier Analysis"
)

plt.grid(True, alpha=0.25)

spectrum_path = (
    OUT_DIR /
    "band08_carrier_spectrum.png"
)

plt.tight_layout()
plt.savefig(
    spectrum_path,
    dpi=150
)

plt.close()

print()
print(f"Saved spectrum: {spectrum_path}")

# ============================================================
# 3. SELECT CANDIDATE CARRIERS
# ============================================================

# We don't blindly assume one carrier.
# Test the strongest few spectral peaks.

candidate_carriers = []

for idx in peak_indices[:8]:

    fc = float(f[idx])

    # Ignore very low-frequency components
    if fc >= 100:
        candidate_carriers.append(fc)

print()
print("Candidate carriers:")

for fc in candidate_carriers:
    print(f"  {fc:.2f} Hz")

# ============================================================
# 4. COHERENT / AM DEMODULATION
# ============================================================

def demodulate_at_carrier(x, fs, fc):

    # --------------------------------------------------------
    # Narrow band-pass around carrier
    # --------------------------------------------------------

    bandwidth = max(
        20.0,
        min(
            80.0,
            fc * 0.15
        )
    )

    low = fc - bandwidth
    high = fc + bandwidth

    if low <= 1:
        low = 1

    if high >= fs / 2:
        high = fs / 2 - 10

    sos = signal.butter(
        4,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    carrier_band = signal.sosfiltfilt(
        sos,
        x
    )

    # --------------------------------------------------------
    # Analytic signal
    # --------------------------------------------------------

    analytic = signal.hilbert(
        carrier_band
    )

    amplitude = np.abs(
        analytic
    )

    phase = np.unwrap(
        np.angle(analytic)
    )

    # --------------------------------------------------------
    # Remove slow amplitude baseline
    # --------------------------------------------------------

    baseline_sos = signal.butter(
        3,
        2.0,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    baseline = signal.sosfiltfilt(
        baseline_sos,
        amplitude
    )

    am = amplitude - baseline

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We do NOT restrict this to 1-20 Hz.
    #
    # Speech information may occupy much higher modulation
    # frequencies.
    # --------------------------------------------------------

    speech_sos = signal.butter(
        4,
        [80, 5000],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    demod = signal.sosfiltfilt(
        speech_sos,
        am
    )

    demod -= np.mean(demod)

    demod /= (
        np.max(np.abs(demod))
        + 1e-12
    )

    return demod, carrier_band, amplitude


# ============================================================
# 5. PROCESS CANDIDATES
# ============================================================

results = []

print()
print("============================================================")
print("TESTING CARRIERS")
print("============================================================")

for fc in candidate_carriers:

    print(
        f"\nTesting carrier: {fc:.2f} Hz"
    )

    try:

        demod, carrier_band, amplitude = (
            demodulate_at_carrier(
                x,
                fs,
                fc
            )
        )

        # ----------------------------------------------------
        # Measure resulting signal
        # ----------------------------------------------------

        rms = np.sqrt(
            np.mean(
                demod ** 2
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        safe_fc = int(round(fc))

        output = (
            OUT_DIR /
            f"band08_demod_{safe_fc}Hz.wav"
        )

        pcm = np.int16(
            np.clip(
                demod * 0.9,
                -1,
                1
            ) * 32767
        )

        wavfile.write(
            str(output),
            fs,
            pcm
        )

        print(
            f"  RMS   : {rms:.4f}"
        )

        print(
            f"  Saved : {output.name}"
        )

        results.append(
            (fc, demod)
        )

    except Exception as e:

        print(
            f"  FAILED: {e}"
        )

# ============================================================
# 6. CREATE SPECTROGRAMS OF DEMODULATED SIGNALS
# ============================================================

print()
print("Creating demodulated spectrograms...")

for fc, demod in results:

    safe_fc = int(round(fc))

    plt.figure(
        figsize=(12, 5)
    )

    plt.specgram(
        demod,
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
        f"Band 08 Demodulated Signal - "
        f"{fc:.1f} Hz Carrier"
    )

    plt.colorbar(
        label="Power"
    )

    output = (
        OUT_DIR /
        f"band08_demod_{safe_fc}Hz_spectrogram.png"
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print(
        f"  {output.name}"
    )

print()
print("============================================================")
print("DONE")
print("============================================================")
print()
print(
    "Listen to the generated band08_demod_*Hz.wav files."
)
print()