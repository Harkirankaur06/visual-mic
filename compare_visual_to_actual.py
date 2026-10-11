from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


# ============================================================
# PATHS
# ============================================================

ACTUAL = Path(
    r"C:\Users\Harkiran\Downloads\actual_video_audio\itsybitsy_original_audio.wav")

VISUAL_ACTUAL = Path(
    r"C:\Users\Harkiran\Downloads\water_surface\water_surface_rs_actual.wav")

VISUAL_MEDIAN = Path(
    r"C:\Users\Harkiran\Downloads\water_surface\water_surface_rs_median.wav")

OUT_DIR = Path(
    r"C:\Users\Harkiran\Downloads\water_surface"
) / "water_surface_comparison"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD WAV
# ============================================================

def load_audio(path):

    fs, x = wavfile.read(
        str(path)
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

    if peak > 0:
        x /= peak

    return fs, x


print()
print("=" * 70)
print("VISUAL vs ACTUAL AUDIO COMPARISON")
print("=" * 70)


fs_real, real = load_audio(
    ACTUAL
)

fs_va, visual_actual = load_audio(
    VISUAL_ACTUAL
)

fs_vm, visual_median = load_audio(
    VISUAL_MEDIAN
)


print()
print(
    f"Actual audio     : {len(real)/fs_real:.3f} s"
)

print(
    f"Visual actual    : {len(visual_actual)/fs_va:.3f} s"
)

print(
    f"Visual median    : {len(visual_median)/fs_vm:.3f} s"
)


# ============================================================
# RESAMPLE ALL TO COMMON RATE
# ============================================================

TARGET_FS = 16000


def resample_audio(x, fs):

    target_len = int(
        len(x) *
        TARGET_FS /
        fs
    )

    return signal.resample(
        x,
        target_len
    )


real = resample_audio(
    real,
    fs_real
)

visual_actual = resample_audio(
    visual_actual,
    fs_va
)

visual_median = resample_audio(
    visual_median,
    fs_vm
)


# ============================================================
# MATCH LENGTH
# ============================================================

N = min(
    len(real),
    len(visual_actual),
    len(visual_median)
)

real = real[:N]
visual_actual = visual_actual[:N]
visual_median = visual_median[:N]


# ============================================================
# SPEECH-BAND FILTER
# ============================================================

def speech_band(x):

    sos = signal.butter(
        4,
        [80, 5000],
        btype="bandpass",
        fs=TARGET_FS,
        output="sos"
    )

    return signal.sosfiltfilt(
        sos,
        x
    )


real_speech = speech_band(
    real
)

visual_actual_speech = speech_band(
    visual_actual
)

visual_median_speech = speech_band(
    visual_median
)


# ============================================================
# RMS ENVELOPE
# ============================================================

def rms_envelope(
    x,
    fs,
    window_ms=40
):

    window = int(
        fs *
        window_ms /
        1000
    )

    window = max(
        window,
        1
    )

    power = signal.fftconvolve(
        x ** 2,
        np.ones(window) / window,
        mode="same"
    )

    env = np.sqrt(
        np.maximum(
            power,
            0
        )
    )

    return env


real_env = rms_envelope(
    real_speech,
    TARGET_FS
)

visual_actual_env = rms_envelope(
    visual_actual_speech,
    TARGET_FS
)

visual_median_env = rms_envelope(
    visual_median_speech,
    TARGET_FS
)


# ============================================================
# NORMALIZE ENVELOPES
# ============================================================

def zscore(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    return (
        x -
        np.mean(x)
    ) / (
        np.std(x) +
        1e-12
    )


real_env_z = zscore(
    real_env
)

visual_actual_env_z = zscore(
    visual_actual_env
)

visual_median_env_z = zscore(
    visual_median_env
)


# ============================================================
# CORRELATION
# ============================================================

corr_actual = np.corrcoef(
    real_env_z,
    visual_actual_env_z
)[0, 1]

corr_median = np.corrcoef(
    real_env_z,
    visual_median_env_z
)[0, 1]


print()
print("=" * 70)
print("ENVELOPE CORRELATION")
print("=" * 70)

print(
    f"Actual vs visual-actual : "
    f"{corr_actual:.4f}"
)

print(
    f"Actual vs visual-median : "
    f"{corr_median:.4f}"
)


# ============================================================
# LAGGED CORRELATION
# ============================================================

# Test whether the visual signal is shifted relative
# to the microphone recording.

MAX_LAG_SECONDS = 2.0

max_lag = int(
    MAX_LAG_SECONDS *
    TARGET_FS
)

# Downsample envelope for efficient lag search
ENV_FS = 100

real_env_ds = signal.resample_poly(
    real_env_z,
    ENV_FS,
    TARGET_FS
)

va_env_ds = signal.resample_poly(
    visual_actual_env_z,
    ENV_FS,
    TARGET_FS
)

vm_env_ds = signal.resample_poly(
    visual_median_env_z,
    ENV_FS,
    TARGET_FS
)

max_lag_ds = int(
    MAX_LAG_SECONDS *
    ENV_FS
)


def best_lag(
    reference,
    candidate
):

    lags = np.arange(
        -max_lag_ds,
        max_lag_ds + 1
    )

    correlations = []

    for lag in lags:

        if lag < 0:

            a = reference[
                :lag
            ]

            b = candidate[
                -lag:
            ]

        elif lag > 0:

            a = reference[
                lag:
            ]

            b = candidate[
                :-lag
            ]

        else:

            a = reference
            b = candidate

        n = min(
            len(a),
            len(b)
        )

        if n < 10:

            correlations.append(
                0
            )

            continue

        a = a[:n]
        b = b[:n]

        c = np.corrcoef(
            a,
            b
        )[0, 1]

        correlations.append(
            c
        )

    correlations = np.asarray(
        correlations
    )

    index = np.argmax(
        correlations
    )

    return (
        lags[index] / ENV_FS,
        correlations[index]
    )


lag_va, lagcorr_va = best_lag(
    real_env_ds,
    va_env_ds
)

lag_vm, lagcorr_vm = best_lag(
    real_env_ds,
    vm_env_ds
)


print()
print("=" * 70)
print("BEST TIME-ALIGNMENT")
print("=" * 70)

print(
    f"Visual actual : "
    f"lag={lag_va:+.3f} s, "
    f"correlation={lagcorr_va:.4f}"
)

print(
    f"Visual median : "
    f"lag={lag_vm:+.3f} s, "
    f"correlation={lagcorr_vm:.4f}"
)


# ============================================================
# PLOT ENVELOPES
# ============================================================

time = np.arange(
    N
) / TARGET_FS

# Downsample only for plotting
plot_step = max(
    1,
    TARGET_FS // 100
)

tp = time[::plot_step]

real_plot = real_env_z[
    ::plot_step
]

va_plot = visual_actual_env_z[
    ::plot_step
]

vm_plot = visual_median_env_z[
    ::plot_step
]


plt.figure(
    figsize=(15, 7)
)

plt.plot(
    tp,
    real_plot,
    label="Actual microphone"
)

plt.plot(
    tp,
    va_plot,
    label="Visual RS actual"
)

plt.plot(
    tp,
    vm_plot,
    label="Visual RS median"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Normalized speech-band RMS"
)

plt.title(
    "Actual vs Visual-Recovered Speech Activity"
)

plt.legend()

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()

envelope_plot = (
    OUT_DIR /
    "01_envelope_comparison.png"
)

plt.savefig(
    envelope_plot,
    dpi=180
)

plt.close()


# ============================================================
# SPECTROGRAM COMPARISON
# ============================================================

signals = [
    (
        real_speech,
        "Actual microphone"
    ),
    (
        visual_actual_speech,
        "Visual RS actual"
    ),
    (
        visual_median_speech,
        "Visual RS median"
    )
]


fig, axes = plt.subplots(
    3,
    1,
    figsize=(15, 12)
)

for ax, (x, title) in zip(
    axes,
    signals
):

    f, t, S = signal.spectrogram(
        x,
        fs=TARGET_FS,
        window="hann",
        nperseg=2048,
        noverlap=1536,
        mode="magnitude"
    )

    Sdb = (
        20 *
        np.log10(
            S + 1e-10
        )
    )

    im = ax.pcolormesh(
        t,
        f,
        Sdb,
        shading="auto"
    )

    ax.set_ylim(
        0,
        5000
    )

    ax.set_ylabel(
        "Hz"
    )

    ax.set_title(
        title
    )

    fig.colorbar(
        im,
        ax=ax
    )

axes[-1].set_xlabel(
    "Time (s)"
)

plt.tight_layout()

spectrogram_plot = (
    OUT_DIR /
    "02_spectrogram_comparison.png"
)

plt.savefig(
    spectrogram_plot,
    dpi=180
)

plt.close()


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

with open(
    OUT_DIR / "comparison_results.txt",
    "w"
) as f:

    f.write(
        "VISUAL vs ACTUAL AUDIO COMPARISON\n"
    )

    f.write(
        "=================================\n\n"
    )

    f.write(
        f"Direct envelope correlation "
        f"(visual actual): {corr_actual:.6f}\n"
    )

    f.write(
        f"Direct envelope correlation "
        f"(visual median): {corr_median:.6f}\n\n"
    )

    f.write(
        f"Best lag visual actual: "
        f"{lag_va:+.3f} s\n"
    )

    f.write(
        f"Best lag correlation visual actual: "
        f"{lagcorr_va:.6f}\n\n"
    )

    f.write(
        f"Best lag visual median: "
        f"{lag_vm:+.3f} s\n"
    )

    f.write(
        f"Best lag correlation visual median: "
        f"{lagcorr_vm:.6f}\n"
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
    "Results:"
)

print(
    OUT_DIR
)

print()
print(
    "Open:"
)

print(
    "  01_envelope_comparison.png"
)

print(
    "  02_spectrogram_comparison.png"
)

print()
print(
    "And send me the terminal output."
)