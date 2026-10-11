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

if not VIDEO.exists():
    raise FileNotFoundError(f"Video not found:\n{VIDEO}")

# Samsung A15 calibrated rolling-shutter delay
ROW_DELAY = 7.7435e-6

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO), cv2.CAP_FFMPEG)

if not cap.isOpened():
    cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open new vid.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video:", W, "x", H)
print("FPS:", fps)
print("Frames:", nframes)

ok, frame0 = cap.read()

if not ok:
    raise RuntimeError("Could not read first frame.")

# ============================================================
# MANUAL SINGLE-U POINT SELECTION
# ============================================================

display = frame0.copy()

cv2.putText(
    display,
    "CLICK 8-20 POINTS ALONG THE SINGLE U-CURVE",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 255),
    2
)

cv2.putText(
    display,
    "Press ENTER when finished | R = reset | ESC = quit",
    (30, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 255, 255),
    2
)

points = []

def mouse_callback(event, x, y, flags, param):

    global display

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append([x, y])

        cv2.circle(
            display,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            display,
            str(len(points)),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

cv2.namedWindow(
    "SELECT SINGLE U CURVE",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "SELECT SINGLE U CURVE",
    900,
    900
)

cv2.setMouseCallback(
    "SELECT SINGLE U CURVE",
    mouse_callback
)

while True:

    cv2.imshow(
        "SELECT SINGLE U CURVE",
        display
    )

    key = cv2.waitKey(20) & 0xFF

    if key == 13:       # ENTER
        break

    if key == ord("r"):

        points = []
        display = frame0.copy()

        cv2.putText(
            display,
            "CLICK 8-20 POINTS ALONG THE SINGLE U-CURVE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "Press ENTER when finished | R = reset | ESC = quit",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    if key == 27:       # ESC
        cv2.destroyAllWindows()
        cap.release()
        raise SystemExit

cv2.destroyWindow(
    "SELECT SINGLE U CURVE"
)

if len(points) < 5:
    cap.release()
    raise RuntimeError(
        "Please select at least 5 points on the U-curve."
    )

p0 = np.asarray(
    points,
    dtype=np.float32
).reshape(-1, 1, 2)

print()
print("Selected U-curve points:", len(points))

# ============================================================
# TRACKING
# ============================================================

lk_params = dict(
    winSize=(31, 31),
    maxLevel=4,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        40,
        0.01
    )
)

prev_gray = cv2.cvtColor(
    frame0,
    cv2.COLOR_BGR2GRAY
)

prev_pts = p0.copy()

records = []

# Frame 0
for i, p in enumerate(prev_pts):

    records.append([
        0,
        i,
        float(p[0,0]),
        float(p[0,1]),
        0.0,
        0.0
    ])

# ============================================================
# OUTPUT OVERLAY VIDEO
# ============================================================

overlay_path = OUT / "single_U_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(overlay_path),
    fourcc,
    fps,
    (W, H)
)

# Frame 0 overlay
overlay = frame0.copy()

for i, p in enumerate(prev_pts):

    x = int(p[0,0])
    y = int(p[0,1])

    cv2.circle(
        overlay,
        (x, y),
        4,
        (0,255,0),
        -1
    )

cv2.putText(
    overlay,
    "SINGLE U-CURVE TRACK",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0,255,0),
    2
)

writer.write(overlay)

# ============================================================
# TRACK FRAMES
# ============================================================

for frame_idx in range(1, nframes):

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

    jump = np.linalg.norm(
        next_pts[:,0,:] -
        prev_pts[:,0,:],
        axis=1
    )

    good = (
        (status == 1) &
        np.isfinite(next_pts[:,0,0]) &
        np.isfinite(next_pts[:,0,1]) &
        np.isfinite(err) &
        (err < 40) &
        (jump < 15)
    )

    for i in np.where(good)[0]:

        x = float(next_pts[i,0,0])
        y = float(next_pts[i,0,1])

        dx = x - float(p0[i,0,0])
        dy = y - float(p0[i,0,1])

        records.append([
            frame_idx,
            i,
            x,
            y,
            dx,
            dy
        ])

    # --------------------------------------------------------
    # DRAW TRACK
    # --------------------------------------------------------

    overlay = frame.copy()

    for i in np.where(good)[0]:

        x = int(next_pts[i,0,0])
        y = int(next_pts[i,0,1])

        cv2.circle(
            overlay,
            (x,y),
            4,
            (0,255,0),
            -1
        )

    cv2.putText(
        overlay,
        f"U points tracked: {np.sum(good)}/{len(p0)}",
        (30,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    writer.write(overlay)

    prev_gray = gray
    prev_pts = next_pts

cap.release()
writer.release()

records = np.asarray(
    records,
    dtype=np.float64
)

tracking_path = OUT / "single_U_tracking.npy"

np.save(
    tracking_path,
    records
)

print()
print("Tracking complete.")
print("Records:", len(records))
print("Tracking video:")
print(overlay_path)

# ============================================================
# AUDIO RECONSTRUCTION
# ============================================================

frame_id = records[:,0]
point_id = records[:,1]
y = records[:,3]
dy = records[:,5]

# ------------------------------------------------------------
# Remove each point's static offset
# ------------------------------------------------------------

for p in np.unique(point_id):

    mask = point_id == p

    if np.sum(mask) > 3:

        dy[mask] -= np.median(
            dy[mask]
        )

# ------------------------------------------------------------
# Rolling-shutter timestamps
# ------------------------------------------------------------

times = (
    frame_id / fps
    +
    y * ROW_DELAY
)

# ------------------------------------------------------------
# Sort temporally
# ------------------------------------------------------------

order = np.argsort(times)

times = times[order]
dy = dy[order]

# ------------------------------------------------------------
# Remove duplicate / non-finite values
# ------------------------------------------------------------

valid = (
    np.isfinite(times) &
    np.isfinite(dy)
)

times = times[valid]
dy = dy[valid]

# ------------------------------------------------------------
# Audio-rate interpolation
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
# Remove slow drift
# ------------------------------------------------------------

audio -= np.mean(audio)

# ------------------------------------------------------------
# Bass band
# ------------------------------------------------------------

sos = butter(
    5,
    [20, 250],
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

audio -= np.mean(audio)

rms = np.sqrt(
    np.mean(audio**2)
)

if rms > 1e-12:
    audio /= rms

# Don't go crazy with gain
audio *= 2.0

audio = np.clip(
    audio,
    -1,
    1
)

wav = np.int16(
    audio * 32767
)

wav_path = OUT / "single_U_bass_recovered.wav"

wavfile.write(
    wav_path,
    FS,
    wav
)

print()
print("==========================================")
print("SINGLE U AUDIO COMPLETE")
print("==========================================")
print("Duration:", len(wav)/FS, "seconds")
print("WAV:")
print(wav_path)
print()
