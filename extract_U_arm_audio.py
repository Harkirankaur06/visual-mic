import cv2
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile

# ============================================================
# PATHS
# ============================================================

VIDEO = Path(r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4")
OUT = Path(r"C:\Users\Harkiran\Downloads\light vibe exp")

OUT.mkdir(parents=True, exist_ok=True)

print("Video path:")
print(VIDEO)
print()

if not VIDEO.exists():
    raise FileNotFoundError(
        f"VIDEO FILE NOT FOUND:\n{VIDEO}"
    )

print("Video file found.")
print()

# ============================================================
# CALIBRATED ROLLING SHUTTER
# ============================================================

ROW_DELAY = 7.7435e-6

# ============================================================
# U-SHAPED TARGET
# ============================================================
#
# Based on your marked U.
#
# We deliberately cover the U and NOT the whole water region.
#

LEFT_BOX  = (180, 770, 265, 920)
RIGHT_BOX = (265, 770, 350, 920)

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():

    # Try OpenCV's FFmpeg backend explicitly
    cap.release()

    cap = cv2.VideoCapture(
        str(VIDEO),
        cv2.CAP_FFMPEG
    )

if not cap.isOpened():
    raise RuntimeError(
        "OpenCV found the file but could not decode it.\n"
        "The file exists, but the video codec could not be opened."
    )

fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video opened successfully.")
print("Resolution:", W, "x", H)
print("FPS:", fps)
print("Frames:", nframes)
print()

# ============================================================
# FIRST FRAME
# ============================================================

ok, frame0 = cap.read()

if not ok:
    raise RuntimeError("Could not read first frame.")

gray0 = cv2.cvtColor(
    frame0,
    cv2.COLOR_BGR2GRAY
)

# ============================================================
# FEATURE DETECTION
# ============================================================

def make_points(gray, box, max_points=50):

    x1, y1, x2, y2 = box

    crop = gray[
        y1:y2,
        x1:x2
    ]

    pts = cv2.goodFeaturesToTrack(
        crop,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=3,
        blockSize=7,
        useHarrisDetector=False
    )

    if pts is None:
        return np.empty(
            (0,1,2),
            dtype=np.float32
        )

    pts[:,0,0] += x1
    pts[:,0,1] += y1

    return pts.astype(np.float32)


left0 = make_points(
    gray0,
    LEFT_BOX,
    60
)

right0 = make_points(
    gray0,
    RIGHT_BOX,
    60
)

print("Left U-arm points :", len(left0))
print("Right U-arm points:", len(right0))

if len(left0) < 3 or len(right0) < 3:
    raise RuntimeError(
        "Not enough feature points detected on one or both U-arms."
    )

p0 = np.vstack([
    left0,
    right0
])

# ============================================================
# LK PARAMETERS
# ============================================================

lk_params = dict(
    winSize=(25,25),
    maxLevel=4,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)

prev_gray = gray0
prev_pts = p0.copy()

records = []

# ============================================================
# OUTPUT TRACKING VIDEO
# ============================================================

overlay_path = OUT / "U_arm_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(overlay_path),
    fourcc,
    fps,
    (W,H)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create tracking output video."
    )

# ============================================================
# FRAME 0
# ============================================================

overlay = frame0.copy()

for i,p in enumerate(prev_pts):

    x = int(p[0,0])
    y = int(p[0,1])

    if i < len(left0):
        c = (0,255,0)
    else:
        c = (255,180,0)

    cv2.circle(
        overlay,
        (x,y),
        3,
        c,
        -1
    )

cv2.rectangle(
    overlay,
    LEFT_BOX[:2],
    LEFT_BOX[2:],
    (0,255,0),
    1
)

cv2.rectangle(
    overlay,
    RIGHT_BOX[:2],
    RIGHT_BOX[2:],
    (255,180,0),
    1
)

writer.write(overlay)

# ============================================================
# TRACK ALL FRAMES
# ============================================================

for frame_idx in range(1,nframes):

    ok, frame = cap.read()

    if not ok:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        None,
        **lk_params
    )

    if next_pts is None:
        break

    status = status.reshape(-1)
    err = err.reshape(-1)

    displacement = (
        next_pts[:,0,:]
        -
        prev_pts[:,0,:]
    )

    jump = np.linalg.norm(
        displacement,
        axis=1
    )

    good = (
        (status == 1)
        &
        np.isfinite(next_pts[:,0,0])
        &
        np.isfinite(next_pts[:,0,1])
        &
        np.isfinite(err)
        &
        (err < 35)
        &
        (jump < 12)
    )

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    for i in np.where(good)[0]:

        x = float(
            next_pts[i,0,0]
        )

        y = float(
            next_pts[i,0,1]
        )

        dx = float(
            x - p0[i,0,0]
        )

        dy = float(
            y - p0[i,0,1]
        )

        records.append([
            frame_idx,
            i,
            x,
            y,
            dx,
            dy
        ])

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    overlay = frame.copy()

    for i in np.where(good)[0]:

        x = int(
            next_pts[i,0,0]
        )

        y = int(
            next_pts[i,0,1]
        )

        if i < len(left0):
            c = (0,255,0)
        else:
            c = (255,180,0)

        cv2.circle(
            overlay,
            (x,y),
            3,
            c,
            -1
        )

    cv2.putText(
        overlay,
        "GREEN = LEFT U ARM",
        (30,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        overlay,
        "ORANGE = RIGHT U ARM",
        (30,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,180,0),
        2
    )

    cv2.putText(
        overlay,
        f"Tracked: {np.sum(good)}",
        (30,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    writer.write(overlay)

    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    prev_gray = gray
    prev_pts = next_pts

cap.release()
writer.release()

records = np.asarray(
    records,
    dtype=np.float64
)

tracking_file = OUT / "U_arm_tracking.npy"

np.save(
    tracking_file,
    records
)

print()
print("==========================================")
print("TRACKING COMPLETE")
print("==========================================")
print("Records:", len(records))
print("Tracking video:")
print(overlay_path)
print()

# ============================================================
# AUDIO RECONSTRUCTION
# ============================================================

frame_id = records[:,0]
y = records[:,3]
dy = records[:,5]

# ------------------------------------------------------------
# Remove common frame-wise motion
# ------------------------------------------------------------

for f in np.unique(frame_id):

    mask = frame_id == f

    if np.sum(mask) > 2:
        dy[mask] -= np.median(
            dy[mask]
        )

# ------------------------------------------------------------
# Rolling-shutter timestamp
# ------------------------------------------------------------

times = (
    frame_id / fps
    +
    y * ROW_DELAY
)

order = np.argsort(times)

times = times[order]
dy = dy[order]

# ------------------------------------------------------------
# Convert to audio-rate signal
# ------------------------------------------------------------

FS = 48000

audio_t = np.arange(
    times[0],
    times[-1],
    1.0 / FS
)

audio = np.interp(
    audio_t,
    times,
    dy
)

# ------------------------------------------------------------
# Remove DC
# ------------------------------------------------------------

audio -= np.mean(audio)

# ------------------------------------------------------------
# Bass filter
# ------------------------------------------------------------

sos = butter(
    5,
    [20,250],
    btype="bandpass",
    fs=FS,
    output="sos"
)

audio = sosfiltfilt(
    sos,
    audio
)

# ------------------------------------------------------------
# Normalize
# ------------------------------------------------------------

rms = np.sqrt(
    np.mean(audio**2)
)

if rms > 1e-12:
    audio /= rms

# Moderate gain
audio *= 2.0

audio = np.clip(
    audio,
    -1,
    1
)

wav = np.int16(
    audio * 32767
)

wav_path = OUT / "U_arm_bass_recovered.wav"

wavfile.write(
    wav_path,
    FS,
    wav
)

print()
print("==========================================")
print("U-ARM AUDIO COMPLETE")
print("==========================================")
print("Duration:", len(wav)/FS, "seconds")
print("WAV:")
print(wav_path)
print()
