import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, coherence

OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")

D = np.load(OUT / "straight_edge_tracking.npz")

profiles = D["profiles"].astype(float)
deformation = D["deformation"].astype(float)
fps = float(D["fps"])

# ------------------------------------------------------------
# 1. Remove spatially static shape
# ------------------------------------------------------------

# Frame-to-frame edge displacement at every x position
motion = profiles - np.median(profiles, axis=0, keepdims=True)

# Remove frame-wide common offset
motion -= np.median(motion, axis=1, keepdims=True)

print("Motion shape:", motion.shape)
print("Motion std:", np.std(motion))

# ------------------------------------------------------------
# 2. Find spatial channels with strongest temporal motion
# ------------------------------------------------------------

channel_std = np.std(motion, axis=0)

idx = np.argsort(channel_std)[::-1]

print()
print("Top moving edge channels:")

for i in idx[:20]:
    print(
        f"x-channel {i:4d}   std={channel_std[i]:.4f}"
    )

# ------------------------------------------------------------
# 3. Build several candidate visual signals
# ------------------------------------------------------------

# A: median of strongest 10% channels
n = max(10, int(0.10 * motion.shape[1]))
strong = idx[:n]

sig_median = np.median(
    motion[:, strong],
    axis=1
)

# B: mean of strongest channels
sig_mean = np.mean(
    motion[:, strong],
    axis=1
)

# C: RMS motion across all channels
sig_rms = np.sqrt(
    np.mean(motion**2, axis=1)
)

# D: PCA-like first spatial mode using SVD
U, S, Vt = np.linalg.svd(
    motion,
    full_matrices=False
)

sig_pca = U[:, 0] * S[0]

signals = {
    "median_top10": sig_median,
    "mean_top10": sig_mean,
    "rms": sig_rms,
    "pca1": sig_pca
}

# ------------------------------------------------------------
# 4. Load CLEAN bass reference
# ------------------------------------------------------------

fs, actual = wavfile.read(
    r"C:\Users\Harkiran\Downloads\light vibe exp\actual_bass_audio.wav"
)

actual = actual.astype(float)

if actual.ndim > 1:
    actual = np.mean(actual, axis=1)

print()
print("Reference:", len(actual), "samples,", fs, "Hz")

# ------------------------------------------------------------
# 5. Filter
# ------------------------------------------------------------

def bandpass(x, fs):

    sos = butter(
        4,
        [20, 250],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, x)

actual_bass = bandpass(actual, fs)

# ------------------------------------------------------------
# 6. Compare each visual candidate
# ------------------------------------------------------------

print()
print("==========================================")
print("VISUAL EDGE / CLEAN BASS COMPARISON")
print("==========================================")

results = {}

for name, sig in signals.items():

    # Interpolate frame-rate signal to audio rate
    t = np.arange(len(sig)) / fps

    duration = min(
        t[-1],
        len(actual_bass) / fs
    )

    ta = np.arange(
        0,
        duration,
        1.0 / fs
    )

    v = np.interp(
        ta,
        t,
        sig
    )

    v = bandpass(v, fs)

    N = min(len(v), len(actual_bass))

    v = v[:N]
    a = actual_bass[:N]

    # normalize
    v -= np.mean(v)
    a -= np.mean(a)

    sv = np.std(v)
    sa = np.std(a)

    if sv < 1e-12:
        corr = np.nan
        coh = np.nan
        cf = np.nan

    else:

        v /= sv
        a /= sa

        # envelope correlation
        win = int(0.05 * fs)
        kernel = np.ones(win) / win

        ve = np.sqrt(
            np.convolve(v*v, kernel, mode="same")
        )

        ae = np.sqrt(
            np.convolve(a*a, kernel, mode="same")
        )

        corr = np.corrcoef(ve, ae)[0,1]

        # coherence
        f, C = coherence(
            a,
            v,
            fs=fs,
            nperseg=min(16384, N)
        )

        mask = (
            (f >= 20) &
            (f <= 250)
        )

        if np.any(mask):
            cf = f[mask][np.argmax(C[mask])]
            coh = np.max(C[mask])
        else:
            cf = np.nan
            coh = np.nan

    results[name] = (
        corr,
        coh,
        cf
    )

    print()
    print(name)
    print("  envelope correlation =", corr)
    print("  maximum coherence    =", coh)
    print("  coherence frequency  =", cf)

# ------------------------------------------------------------
# 7. Save raw candidate signals
# ------------------------------------------------------------

np.savez(
    OUT / "edge_motion_candidates.npz",
    motion=motion,
    channel_std=channel_std,
    median_top10=sig_median,
    mean_top10=sig_mean,
    rms=sig_rms,
    pca1=sig_pca,
    fps=fps
)

print()
print("Saved:")
print(OUT / "edge_motion_candidates.npz")
