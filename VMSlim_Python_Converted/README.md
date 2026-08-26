# VMSlim — Python conversion

This is a Python conversion of the VMSlim / Visual Microphone MATLAB code supplied in `VMSlim.zip`.

## What is included

`visual_microphone.py` contains:

- `vmSoundFromVideo.m`
- `buildSCFpyr.m`
- `buildSCFpyrLevs.m`
- `vmAlignAToB.m`
- `vmGetSoundScaledToOne.m`
- `vmGetSoundSpecSub.m`
- `vmComputeSTFT.m`
- `vmComputeSpecSub.m`
- `vmSTFTForward.m`
- `vmSTFTResynth.m`
- `vmWriteWAV.m`

The MATLAB `matlabPyrTools` dependency is replaced by a direct Python implementation of the complex frequency-domain steerable pyramid used by the supplied code. This means the Python program does not require MATLAB or the `.mexw64` files.

## Install

Create a virtual environment if desired:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

## Run

The supplied MATLAB demo uses:

- 1 scale
- 2 orientations
- 0.1 spatial downsampling
- 2200 Hz output sampling rate

Equivalent Python command:

```bash
python visual_microphone.py crabchipsRamp.avi --output RecoveredSound.wav --sampling-rate 2200 --scales 1 --orientations 2 --downsample 0.1
```

If you want to process only the first 500 frames:

```bash
python visual_microphone.py crabchipsRamp.avi --frames 500
```

To also perform the MATLAB spectral-subtraction step:

```bash
python visual_microphone.py crabchipsRamp.avi --spectral-subtraction
```

This creates:

```text
RecoveredSound.wav
RecoveredSound_specsub.wav
```

## Important

The input video should contain visible tiny motions/vibrations caused by sound. This is not ordinary audio extraction from a video file: the algorithm estimates sound from sub-pixel visual motion.

The original MATLAB demo manually sets the output sampling rate to 2200 Hz. The Python version keeps that behavior by default.

The Python version uses OpenCV for video decoding, NumPy/SciPy for numerical processing, and SoundFile for WAV output.

## MATLAB → Python correspondence

| MATLAB | Python |
|---|---|
| `VideoReader` | `cv2.VideoCapture` |
| `imresize` | OpenCV bicubic resize |
| `im2single` | float32 conversion to [0,1] |
| `buildSCFpyr` | `build_scf_pyr` |
| `pyrBand` | dictionary lookup `(level, band)` |
| `angle`, `abs` | `np.angle`, `np.abs` |
| `circshift` | `np.roll` |
| `butter` | `scipy.signal.butter` |
| `filter` | `scipy.signal.lfilter` |
| `spectrogram` | can be added with SciPy/matplotlib |
| `wavwrite` | `soundfile.write` |

## Validation

The included implementation was run on a synthetic video during conversion to verify that the complete video → pyramid → phase → alignment → high-pass → normalization → WAV pipeline executes successfully.
