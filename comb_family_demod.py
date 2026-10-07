import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
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
    "family_demodulation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

FS_AUDIO = 48000

# Strong observed carrier-family regions
CARRIERS = [
    306.286,
    456.306,
    486.282
]

# We will use a BROAD complex band around each carrier.
# This is intentionally much wider than the previous attempt.
BANDWIDTH = 3500

# Output speech range
LOW = 80
HIGH = 5000


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
print("CARRIER-FAMILY COMPLEX DEMODULATION")
print("=" * 70)
print(f"Input     : {INPUT}")
print(f"Sample Hz : {fs}")
print(f"Samples   : {len(x)}")
print(f"Duration  : {len(x)/fs:.3f} s")
print()
print("Carrier family:")
for c in CARRIERS:
    print(f"  {c:.3f} Hz")
print()
print(f"Complex bandwidth : {BANDWIDTH} Hz")
print(f"Output band       : {LOW}-{HIGH} Hz")
print("=" * 70)
print()


# ============================================================
# TIME AXIS
# ============================================================

t = np.arange(
    len(x)
) / fs


# ============================================================
# NORMALIZATION HELPER
# ============================================================

def normalize(y):

    y = np.asarray(
        y,
        dtype=np.float64
    )

    y -= np.mean(y)

    p = np.max(
        np.abs(y)
    )

    if p > 1e-12:
        y /= p

    return y


# ============================================================
# COMPLEX DEMODULATION
# ============================================================

def demodulate(
    x,
    fs,
    carrier
):

    print()
    print(
        f"Testing carrier "
        f"{carrier:.3f} Hz"
    )

    # --------------------------------------------------------
    # 1. Complex analytic signal
    # --------------------------------------------------------

    analytic = signal.hilbert(
        x
    )

    # --------------------------------------------------------
    # 2. Shift carrier to DC
    # --------------------------------------------------------

    mixed = (
        analytic *
        np.exp(
            -2j *
            np.pi *
            carrier *
            t
        )
    )

    # --------------------------------------------------------
    # 3. Keep a VERY broad modulation bandwidth
    # --------------------------------------------------------

    # First lowpass at 5 kHz.
    #
    # This is NOT the final speech extraction.
    # It lets us inspect what actually exists around the carrier.

    sos = signal.butter(
        6,
        HIGH,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    baseband = signal.sosfiltfilt(
        sos,
        mixed
    )

    # --------------------------------------------------------
    # 4. Test different interpretations
    # --------------------------------------------------------

    # A: real part
    real_part = np.real(
        baseband
    )

    # B: imaginary part
    imag_part = np.imag(
        baseband
    )

    # C: magnitude/envelope
    envelope = np.abs(
        baseband
    )

    # D: unwrapped phase
    phase = np.unwrap(
        np.angle(
            baseband
        )
    )

    # --------------------------------------------------------
    # 5. Speech-band filter
    # --------------------------------------------------------

    speech_sos = signal.butter(
        4,
        [LOW, HIGH],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    real_speech = signal.sosfiltfilt(
        speech_sos,
        real_part
    )

    imag_speech = signal.sosfiltfilt(
        speech_sos,
        imag_part
    )

    # Envelope is intentionally filtered more gently.
    envelope_sos = signal.butter(
        4,
        [80, 5000],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    envelope_speech = signal.sosfiltfilt(
        envelope_sos,
        envelope
    )

    # Phase derivative gives instantaneous frequency
    phase_diff = np.gradient(
        phase
    ) * fs / (
        2 * np.pi
    )

    phase_speech = signal.sosfiltfilt(
        speech_sos,
        phase_diff
    )

    # --------------------------------------------------------
    # 6. Normalize
    # --------------------------------------------------------

    outputs = {
        "real": normalize(
            real_speech
        ),
        "imag": normalize(
            imag_speech
        ),
        "envelope": normalize(
            envelope_speech
        ),
        "phase": normalize(
            phase_speech
        )
    }

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    for name, y in outputs.items():

        path = (
            OUT_DIR /
            f"demod_{int(round(carrier))}"
            f"Hz_{name}.wav"
        )

        pcm = np.int16(
            np.clip(
                y * 0.9,
                -1,
                1
            ) * 32767
        )

        wavfile.write(
            str(path),
            fs,
            pcm
        )

        rms = np.sqrt(
            np.mean(
                y ** 2
            )
        )

        print(
            f"  {name:9s} "
            f"RMS={rms:.5f} "
            f"-> {path.name}"
        )

    # --------------------------------------------------------
    # 8. Spectrograms
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 12)
    )

    for ax, (name, y) in zip(
        axes,
        outputs.items()
    ):

        Pxx, freqs, times, im = (
            ax.specgram(
                y,
                NFFT=4096,
                Fs=fs,
                noverlap=3072
            )
        )

        ax.set_ylim(
            0,
            5000
        )

        ax.set_title(
            f"{name} — "
            f"{carrier:.1f} Hz carrier"
        )

        ax.set_ylabel(
            "Hz"
        )

    axes[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        f"Carrier-Family Demodulation "
        f"{carrier:.1f} Hz",
        fontsize=15
    )

    fig.tight_layout()

    fig_path = (
        OUT_DIR /
        f"demod_{int(round(carrier))}"
        f"Hz_all_spectrograms.png"
    )

    fig.savefig(
        fig_path,
        dpi=160
    )

    plt.close(fig)

    print(
        f"  Spectrogram: "
        f"{fig_path.name}"
    )


# ============================================================
# RUN
# ============================================================

for carrier in CARRIERS:

    demodulate(
        x,
        fs,
        carrier
    )


# ============================================================
# ALSO TEST A COMB-CENTERED CARRIER
# ============================================================

# The strongest family is approximately:
#
# 96.316, 126.291, 156.267, ...
#
# These correspond to approximately
# 6.3 + n*30 Hz.
#
# Instead of picking one member, construct a signal from
# ALL family components below 5 kHz.

print()
print("=" * 70)
print("BUILDING 30-HZ FAMILY SIGNAL")
print("=" * 70)


X = np.fft.rfft(
    x
)

freq = np.fft.rfftfreq(
    len(x),
    1 / fs
)

family_mask = np.zeros(
    len(freq),
    dtype=bool
)

# Use the two measured families.
for residue in [6.3, 23.7]:

    n_min = int(
        np.floor(
            (20 - residue) /
            30.000816
        )
    )

    n_max = int(
        np.ceil(
            (5000 - residue) /
            30.000816
        )
    )

    for n in range(
        max(0, n_min),
        n_max + 1
    ):

        center = (
            residue +
            n * 30.000816
        )

        band = (
            np.abs(
                freq - center
            ) < 2.0
        )

        family_mask |= band


X_family = np.zeros_like(
    X
)

X_family[
    family_mask
] = X[
    family_mask
]

family_signal = np.fft.irfft(
    X_family,
    n=len(x)
)

family_signal = normalize(
    family_signal
)

family_path = (
    OUT_DIR /
    "30hz_family_only.wav"
)

wavfile.write(
    str(family_path),
    fs,
    np.int16(
        np.clip(
            family_signal * 0.9,
            -1,
            1
        ) * 32767
    )
)

print(
    f"Saved: {family_path}"
)


# ============================================================
# FAMILY SPECTROGRAM
# ============================================================

plt.figure(
    figsize=(13, 5)
)

plt.specgram(
    family_signal,
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
    "30-Hz Carrier Family Only"
)

plt.colorbar(
    label="Power"
)

plt.tight_layout()

family_spec = (
    OUT_DIR /
    "30hz_family_only_spectrogram.png"
)

plt.savefig(
    family_spec,
    dpi=160
)

plt.close()

print(
    f"Saved: {family_spec}"
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print()
print(
    f"Results:\n{OUT_DIR}"
)
print()
print(
    "FIRST LISTEN TO:"
)
print(
    "  demod_456Hz_real.wav"
)
print(
    "  demod_456Hz_imag.wav"
)
print(
    "  demod_456Hz_envelope.wav"
)
print(
    "  demod_456Hz_phase.wav"
)
print()
print(
    "THEN:"
)
print(
    "  demod_486Hz_*.wav"
)
print()
print(
    "FINALLY:"
)
print(
    "  30hz_family_only.wav"
)
print()