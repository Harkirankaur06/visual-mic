import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT = Path(
    r"C:\Users\Harkiran\Downloads\speech_hybrid"
    r"\hybrid results\candidate_08_band_06.wav"
)

OUT_DIR = INPUT.parent / "sideband_analysis"
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FS_OUT = 48000

# Candidate carriers observed previously
CARRIERS = [
    306.34,
    456.30,
    486.33
]

# Frequency range to inspect
MIN_FREQ = 20
MAX_FREQ = 6000


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

x /= (
    np.max(
        np.abs(x)
    ) + 1e-12
)

print()
print("=" * 70)
print("CARRIER / SIDEBAND ANALYSIS")
print("=" * 70)
print(f"Input     : {INPUT}")
print(f"Sample Hz : {fs}")
print(f"Samples   : {len(x)}")
print(f"Duration  : {len(x)/fs:.3f} s")
print("=" * 70)
print()


# ============================================================
# HIGH-RESOLUTION SPECTRUM
# ============================================================

# Long FFT gives much better frequency resolution
nfft = min(
    1048576,
    len(x)
)

window = signal.windows.hann(
    nfft
)

segment = x[:nfft]

spectrum = np.fft.rfft(
    segment * window,
    n=nfft
)

freq = np.fft.rfftfreq(
    nfft,
    1 / fs
)

power = np.abs(
    spectrum
) ** 2

# Normalize
power /= (
    np.max(power) + 1e-20
)


# ============================================================
# OVERALL SPECTRUM
# ============================================================

mask = (
    (freq >= MIN_FREQ) &
    (freq <= MAX_FREQ)
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    freq[mask],
    10 * np.log10(
        power[mask] + 1e-14
    )
)

for fc in CARRIERS:

    plt.axvline(
        fc,
        linestyle="--",
        label=f"{fc:.1f} Hz"
    )

plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Relative power (dB)"
)

plt.title(
    "Band 08 — Wideband Spectrum"
)

plt.legend()

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

overall_path = (
    OUT_DIR /
    "band08_wideband_spectrum.png"
)

plt.savefig(
    overall_path,
    dpi=180
)

plt.close()

print(
    f"Saved: {overall_path}"
)


# ============================================================
# FIND LOCAL PEAKS
# ============================================================

print()
print("=" * 70)
print("LOCAL SPECTRAL PEAKS")
print("=" * 70)

peaks, properties = signal.find_peaks(
    power[mask],
    prominence=1e-5,
    distance=10
)

real_indices = np.where(mask)[0][peaks]

real_indices = real_indices[
    np.argsort(
        power[real_indices]
    )[::-1]
]

for i, idx in enumerate(
    real_indices[:50],
    start=1
):

    print(
        f"{i:02d}. "
        f"{freq[idx]:9.3f} Hz   "
        f"{10*np.log10(power[idx]+1e-14):8.2f} dB"
    )


# ============================================================
# ANALYZE EACH CARRIER
# ============================================================

for fc in CARRIERS:

    print()
    print("=" * 70)
    print(
        f"CARRIER {fc:.2f} Hz"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Examine +/- 5000 Hz around carrier
    # --------------------------------------------------------

    local_low = max(
        20,
        fc - 5000
    )

    local_high = min(
        fs / 2 - 1,
        fc + 5000
    )

    local_mask = (
        (freq >= local_low) &
        (freq <= local_high)
    )

    local_freq = freq[
        local_mask
    ]

    local_power = power[
        local_mask
    ]

    # --------------------------------------------------------
    # Plot local spectrum
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        local_freq,
        10 * np.log10(
            local_power + 1e-14
        )
    )

    plt.axvline(
        fc,
        linestyle="--",
        linewidth=2,
        label=f"Carrier {fc:.2f} Hz"
    )

    # Mark expected speech sideband zones
    #
    # Carrier +/- 80 Hz
    # Carrier +/- 300 Hz
    # Carrier +/- 1000 Hz
    # Carrier +/- 3000 Hz

    for offset in [
        80,
        300,
        500,
        1000,
        2000,
        3000,
        4000
    ]:

        for sign in [-1, 1]:

            fmark = fc + sign * offset

            if (
                fmark >= 20 and
                fmark <= fs / 2
            ):

                plt.axvline(
                    fmark,
                    linestyle=":",
                    alpha=0.35
                )

    plt.xlim(
        20,
        min(
            6000,
            fs / 2
        )
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Relative power (dB)"
    )

    plt.title(
        f"Band 08 Around Candidate Carrier "
        f"{fc:.2f} Hz"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        OUT_DIR /
        f"carrier_{int(round(fc))}"
        f"_wideband.png"
    )

    plt.savefig(
        path,
        dpi=180
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    # --------------------------------------------------------
    # HIGH RESOLUTION ZOOM
    # --------------------------------------------------------

    zoom_low = max(
        20,
        fc - 1000
    )

    zoom_high = min(
        fs / 2 - 1,
        fc + 3000
    )

    zoom_mask = (
        (freq >= zoom_low) &
        (freq <= zoom_high)
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        freq[zoom_mask],
        10 * np.log10(
            power[zoom_mask] + 1e-14
        )
    )

    plt.axvline(
        fc,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Relative power (dB)"
    )

    plt.title(
        f"High Resolution Carrier Region "
        f"{fc:.2f} Hz"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    path = (
        OUT_DIR /
        f"carrier_{int(round(fc))}"
        f"_zoom.png"
    )

    plt.savefig(
        path,
        dpi=180
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    # --------------------------------------------------------
    # PRINT SIDEBAND PEAKS
    # --------------------------------------------------------

    # Search +/- 1000 Hz from carrier
    sideband_mask = (
        (freq >= max(20, fc - 1000)) &
        (freq <= fc + 3000)
    )

    sb_indices = np.where(
        sideband_mask
    )[0]

    sb_power = power[
        sb_indices
    ]

    sb_peaks, _ = signal.find_peaks(
        sb_power,
        prominence=1e-5,
        distance=20
    )

    sb_indices = sb_indices[
        sb_peaks
    ]

    sb_indices = sb_indices[
        np.argsort(
            power[sb_indices]
        )[::-1]
    ]

    print()
    print(
        "Strong local components:"
    )

    for idx in sb_indices[:25]:

        offset = (
            freq[idx] - fc
        )

        print(
            f"  {freq[idx]:9.3f} Hz "
            f"offset={offset:+9.3f} Hz "
            f"power="
            f"{10*np.log10(power[idx]+1e-14):8.2f} dB"
        )


# ============================================================
# SPEECH-SCALE SIDE-BAND ENERGY TEST
# ============================================================

print()
print("=" * 70)
print("SIDEBAND ENERGY TEST")
print("=" * 70)

for fc in CARRIERS:

    print()
    print(
        f"Carrier: {fc:.2f} Hz"
    )

    # For several modulation frequencies,
    # compare energy near fc +/- fm.

    for fm in [
        100,
        200,
        300,
        500,
        800,
        1000,
        1500,
        2000,
        3000
    ]:

        lower = abs(
            fc - fm
        )

        upper = (
            fc + fm
        )

        bw = 15

        lower_mask = (
            (freq >= lower - bw) &
            (freq <= lower + bw)
        )

        upper_mask = (
            (freq >= upper - bw) &
            (freq <= upper + bw)
        )

        lower_energy = np.mean(
            power[lower_mask]
        ) if np.any(lower_mask) else 0

        upper_energy = np.mean(
            power[upper_mask]
        ) if np.any(upper_mask) else 0

        print(
            f"  modulation "
            f"{fm:4d} Hz: "
            f"lower={lower_energy:.3e} "
            f"upper={upper_energy:.3e}"
        )


print()
print("=" * 70)
print("DONE")
print("=" * 70)
print()
print(
    f"Results:\n{OUT_DIR}"
)