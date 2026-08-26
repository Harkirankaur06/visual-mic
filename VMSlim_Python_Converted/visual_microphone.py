import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt
from scipy.io import wavfile


# -----------------------------------------------------------------------------
# MATLAB matlabPyrTools-compatible helpers used by buildSCFpyr
# -----------------------------------------------------------------------------

def rcos_fn(width=1.0, position=0.0, values=(0.0, 1.0)):
    sz = 256
    x = np.pi * np.arange(-sz - 1, 2, dtype=np.float64) / (2.0 * sz)
    y = values[0] + (values[1] - values[0]) * np.cos(x) ** 2
    y[0] = y[1]
    y[-1] = y[-2]
    x = position + (2.0 * width / np.pi) * (x + np.pi / 4.0)
    return x, y


def point_op(im, lut, origin, increment):
    """Equivalent to matlabPyrTools pointOp with linear interpolation."""
    x = origin + increment * np.arange(len(lut), dtype=np.float64)
    flat = np.asarray(im, dtype=np.float64).ravel()
    # np.interp clamps; MATLAB pointOp extrapolates linearly, so explicitly
    # extrapolate the two ends.
    out = np.interp(flat, x, lut)
    left = flat < x[0]
    right = flat > x[-1]
    if np.any(left):
        slope = (lut[1] - lut[0]) / (x[1] - x[0])
        out[left] = lut[0] + slope * (flat[left] - x[0])
    if np.any(right):
        slope = (lut[-1] - lut[-2]) / (x[-1] - x[-2])
        out[right] = lut[-1] + slope * (flat[right] - x[-1])
    return out.reshape(np.asarray(im).shape).astype(np.float64)


def build_scfpyr(image, height=1, order=1):
    """Python implementation of matlabPyrTools buildSCFpyr.

    This follows the MATLAB source included with VMSlim rather than using a
    different motion/audio algorithm.

    Returns a list of complex bands and their shapes. Band 0 is the highpass;
    the following bands are the oriented complex bands; the last band is the
    lowpass residual.
    """
    im = np.asarray(image, dtype=np.float64)
    h, w = im.shape
    nbands = order + 1

    # MATLAB max_ht = floor(log2(min(size(im)))) - 2
    max_ht = int(np.floor(np.log2(min(h, w))) - 2)
    if height > max_ht:
        raise ValueError(f"Image {w}x{h} is too small for pyramid height {height}; max is {max_ht}.")

    if nbands % 2 == 0:
        harmonics = np.arange(nbands // 2) * 2 + 1
    else:
        harmonics = np.arange((nbands - 1) // 2 + 1) * 2

    # MATLAB coordinate convention.
    ctr_y = int(np.ceil((h + 0.5) / 2.0))
    ctr_x = int(np.ceil((w + 0.5) / 2.0))
    # Convert MATLAB 1-based coordinate ctr to zero-based indices.
    cy = ctr_y - 1
    cx = ctr_x - 1

    yr, xr = np.meshgrid(
        (np.arange(1, w + 1) - ctr_x) / (w / 2.0),
        (np.arange(1, h + 1) - ctr_y) / (h / 2.0),
    )
    angle = np.arctan2(yr, xr)
    log_rad = np.sqrt(xr ** 2 + yr ** 2)
    if w > 1:
        log_rad[cy, cx] = log_rad[cy, max(cx - 1, 0)]
    log_rad = np.log2(log_rad)

    xrcos, yrcos = rcos_fn(1.0, -0.5, (0.0, 1.0))
    yrcos = np.sqrt(yrcos)
    yi_rcos = np.sqrt(np.maximum(0.0, 1.0 - yrcos ** 2))

    lo0mask = point_op(log_rad, yi_rcos, xrcos[0], xrcos[1] - xrcos[0])
    imdft = np.fft.fftshift(np.fft.fft2(im))
    lo0dft = imdft * lo0mask

    bands, shapes = _build_scfpyr_levs(
        lo0dft, log_rad, xrcos, yrcos, angle, height, nbands
    )

    hi0mask = point_op(log_rad, yrcos, xrcos[0], xrcos[1] - xrcos[0])
    hi0dft = imdft * hi0mask
    hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft)).real

    return [hi0.astype(np.float64)] + bands, [hi0.shape] + shapes


def _build_scfpyr_levs(lodft, log_rad, xrcos, yrcos, angle, ht, nbands):
    if ht <= 0:
        lo0 = np.fft.ifft2(np.fft.ifftshift(lodft)).real
        return [lo0.astype(np.float64)], [lo0.shape]

    lutsize = 1024
    xcosn = np.pi * np.arange(-(2 * lutsize + 1), lutsize + 2) / lutsize
    order = nbands - 1
    const = ((2.0 ** (2 * order)) * (math.factorial(order) ** 2) /
             (nbands * math.factorial(2 * order)))
    alfa = (np.pi + xcosn) % (2.0 * np.pi) - np.pi
    ycosn = 2.0 * np.sqrt(const) * (np.cos(xcosn) ** order) * (np.abs(alfa) < np.pi / 2.0)

    xrcos_shift = xrcos - np.log2(2.0)
    himask = point_op(log_rad, yrcos, xrcos_shift[0], xrcos_shift[1] - xrcos_shift[0])

    bands = []
    shapes = []
    for b in range(nbands):
        anglemask = point_op(
            angle,
            ycosn,
            xcosn[0] + np.pi * b / nbands,
            xcosn[1] - xcosn[0],
        )
        banddft = ((-1j) ** (nbands - 1)) * lodft * anglemask * himask
        band = np.fft.ifft2(np.fft.ifftshift(banddft))
        bands.append(band.astype(np.complex128))
        shapes.append(band.shape)

    dims = lodft.shape
    ctr_y = int(np.ceil((dims[0] + 0.5) / 2.0))
    ctr_x = int(np.ceil((dims[1] + 0.5) / 2.0))
    # MATLAB: lodims = ceil((dims-0.5)/2)
    lodims_y = int(np.ceil((dims[0] - 0.5) / 2.0))
    lodims_x = int(np.ceil((dims[1] - 0.5) / 2.0))
    loctr_y = int(np.ceil((lodims_y + 0.5) / 2.0))
    loctr_x = int(np.ceil((lodims_x + 0.5) / 2.0))
    lostart_y = ctr_y - loctr_y  # zero-based start
    lostart_x = ctr_x - loctr_x
    loend_y = lostart_y + lodims_y
    loend_x = lostart_x + lodims_x

    log_rad2 = log_rad[lostart_y:loend_y, lostart_x:loend_x]
    angle2 = angle[lostart_y:loend_y, lostart_x:loend_x]
    lodft2 = lodft[lostart_y:loend_y, lostart_x:loend_x]

    yi_rcos = np.abs(np.sqrt(np.maximum(0.0, 1.0 - yrcos ** 2)))
    lomask = point_op(log_rad2, yi_rcos, xrcos_shift[0], xrcos_shift[1] - xrcos_shift[0])
    lodft2 = lomask * lodft2

    low_bands, low_shapes = _build_scfpyr_levs(
        lodft2, log_rad2, xrcos, yrcos, angle2, ht - 1, nbands
    )
    return bands + low_bands, shapes + low_shapes


# -----------------------------------------------------------------------------
# VMSlim audio extraction
# -----------------------------------------------------------------------------

def wrapped_phase_difference(current, reference):
    """MATLAB: mod(pi + angle(pyr)-angle(pyrRef), 2*pi)-pi."""
    return (np.pi + np.angle(current) - np.angle(reference)) % (2.0 * np.pi) - np.pi


def align_a_to_b(a, b):
    """Equivalent to vmAlignAToB.m."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    corr = np.convolve(a, b[::-1], mode="full")
    maxind = int(np.argmax(corr))
    shift = len(b) - maxind - 1
    return np.roll(a, shift), shift


def extract_phase_signals(video_path, nscales=1, norientations=2,
                          downsample_factor=0.1, max_frames=0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames and max_frames > 0:
        total = min(total, max_frames)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read the first frame.")

    def preprocess(fr):
        if downsample_factor != 1.0:
            fr = cv2.resize(fr, None, fx=downsample_factor, fy=downsample_factor,
                            interpolation=cv2.INTER_AREA)
        # MATLAB squeeze(mean(colorframe,3))
        return fr.astype(np.float64).mean(axis=2) / 255.0

    ref = preprocess(frame)
    ref_pyr, _ = build_scfpyr(ref, nscales, norientations - 1)

    # For nscales=1/norientations=2, this is [highpass, band0, band1, lowpass].
    # MATLAB selects the oriented bands only, not highpass/lowpass.
    band_offset = 1
    signalffs = np.zeros((nscales, norientations, total), dtype=np.float64)

    # Frame 1 in MATLAB is included and gives phase=0.
    current_frame = frame
    for q in range(total):
        if q > 0:
            ok, current_frame = cap.read()
            if not ok:
                signalffs = signalffs[:, :, :q]
                total = q
                break

        im = preprocess(current_frame)
        pyr, _ = build_scfpyr(im, nscales, norientations - 1)

        for j in range(nscales):
            for k in range(norientations):
                band_idx = band_offset + j * norientations + k
                band = pyr[band_idx]
                ref_band = ref_pyr[band_idx]
                amp = np.abs(band)
                phase = wrapped_phase_difference(band, ref_band)
                phasew = phase * (np.abs(amp) ** 2)
                sumamp = np.sum(np.abs(amp))
                signalffs[j, k, q] = np.mean(phasew) / (sumamp + 1e-12)

        if (q + 1) % max(1, total // 10) == 0 or q == total - 1:
            print(f"Extracting phase: {q + 1}/{total}")

    cap.release()
    return signalffs, fps


def vmslim_recover(signalffs, fps, output_sample_rate=None):
    nscales, norientations, nframes = signalffs.shape

    # MATLAB aligns every scale/orientation signal to signalffs(1,1,:).
    reference = signalffs[0, 0, :]
    aligned = np.zeros((nscales, norientations, nframes), dtype=np.float64)
    for j in range(nscales):
        for k in range(norientations):
            aligned[j, k], _ = align_a_to_b(signalffs[j, k, :], reference)

    sig_aligned = np.sum(aligned, axis=(0, 1))
    average_no_alignment = np.mean(signalffs.reshape(nscales * norientations, nframes), axis=0)

    # MATLAB: butter(3, 0.05, 'high') -> normalized cutoff 0.05 of Nyquist.
    b, a = butter(3, 0.05, btype="high")
    filtered = filtfilt(b, a, sig_aligned) if nframes > 24 else sig_aligned.copy()

    # MATLAB fixes the first 10 entries after filtering.
    if len(filtered) >= 10:
        filtered[:10] = np.mean(filtered)

    maxsx = float(np.max(filtered))
    minsx = float(np.min(filtered))
    if maxsx != 1.0 or minsx != -1.0:
        rng = maxsx - minsx
        if rng > 1e-12:
            filtered = 2.0 * filtered / rng
            filtered -= np.max(filtered) - 1.0

    # The phase sequence is sampled once per video frame. If an integer WAV
    # rate is requested, resample without changing the time duration.
    native_rate = float(fps)
    if output_sample_rate is None:
        output_sample_rate = int(round(native_rate))

    if output_sample_rate != int(round(native_rate)) or abs(output_sample_rate - native_rate) > 1e-9:
        duration = len(filtered) / native_rate
        new_n = max(1, int(round(duration * output_sample_rate)))
        old_t = np.arange(len(filtered), dtype=np.float64) / native_rate
        new_t = np.arange(new_n, dtype=np.float64) / output_sample_rate
        new_t = np.minimum(new_t, old_t[-1] if len(old_t) else 0.0)
        audio = np.interp(new_t, old_t, filtered)
    else:
        audio = filtered

    return audio.astype(np.float64), average_no_alignment, sig_aligned


def write_wav(path, audio, sample_rate):
    x = np.clip(audio, -1.0, 1.0)
    wavfile.write(str(path), int(sample_rate), np.int16(np.round(x * 32767.0)))


def main():
    parser = argparse.ArgumentParser(
        description="Extract a VMSlim/Visual Microphone phase signal from an amplified video."
    )
    parser.add_argument("video", help="Amplified input video")
    parser.add_argument("-o", "--output", default=None, help="Output WAV path")
    parser.add_argument("--downsample", type=float, default=0.1,
                        help="Spatial downsample factor (MATLAB default: 0.1)")
    parser.add_argument("--scales", type=int, default=1)
    parser.add_argument("--orientations", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="WAV sample rate. Default: rounded video FPS.")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    video = Path(args.video)
    output = Path(args.output) if args.output else video.with_name(video.stem + "_recovered.wav")

    print(f"Video: {video}")
    signalffs, fps = extract_phase_signals(
        video,
        nscales=args.scales,
        norientations=args.orientations,
        downsample_factor=args.downsample,
        max_frames=args.max_frames,
    )

    print(f"Video FPS: {fps:.6f}")
    print(f"Frames used: {signalffs.shape[-1]}")
    print(f"Video duration: {signalffs.shape[-1] / fps:.4f} s")

    audio, _, _ = vmslim_recover(
        signalffs, fps, output_sample_rate=args.sample_rate
    )

    sample_rate = args.sample_rate if args.sample_rate is not None else int(round(fps))
    write_wav(output, audio, sample_rate)
    print(f"Recovered WAV: {output}")
    print(f"Audio samples: {len(audio)}")
    print(f"Audio duration: {len(audio) / sample_rate:.4f} s")


if __name__ == "__main__":
    main()
