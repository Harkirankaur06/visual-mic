import cv2
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile

VIDEO = Path(r"C:\Users\Harkiran\Downloads\Mobile Devices\light.mp4")
OUT = Path(r"C:\Users\Harkiran\Downloads\newlightvibeexp")
OUT.mkdir(parents=True, exist_ok=True)

ROW_DELAY = 7.7435e-6

cap = cv2.VideoCapture(str(VIDEO), cv2.CAP_FFMPEG)

if not cap.isOpened():
    cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open light.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video:", W, "x", H)
print("FPS:", fps)
print("Frames:", nframes)

ok, frame0 = cap.read()

if not ok:
    raise RuntimeError("Could not read first frame")

gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

# ============================================================
# SELECT POINTS ON THE HORIZONTAL LIGHT LINE
# ============================================================

display = frame0.copy()
points = []

cv2.putText(
    display,
    "CLICK POINTS ON THE FLUCTUATING HORIZONTAL LINE",
    (30,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0,255,255),
    2
)

cv2.putText(
    display,
    "ENTER = continue | R = reset | ESC = quit",
    (30,70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0,255,255),
    2
)

def mouse(event, x, y, flags, param):

    global display

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append([x,y])

        cv2.circle(
            display,
            (x,y),
            5,
            (0,255,0),
            -1
        )

        cv2.putText(
            display,
            str(len(points)),
            (x+6,y-6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0,255,0),
            1
        )

cv2.namedWindow(
    "SELECT LINE",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "SELECT LINE",
    900,
    900
)

cv2.setMouseCallback(
    "SELECT LINE",
    mouse
)

while True:

    cv2.imshow(
        "SELECT LINE",
        display
    )

    key = cv2.waitKey(20) & 0xFF

    if key == 13:
        break

    if key == ord("r"):

        points = []
        display = frame0.copy()

        cv2.putText(
            display,
            "CLICK POINTS ON THE FLUCTUATING HORIZONTAL LINE",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0,255,255),
            2
        )

        cv2.putText(
            display,
            "ENTER = continue | R = reset | ESC = quit",
            (30,70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,255,255),
            2
        )

    if key == 27:

        cv2.destroyAllWindows()
        cap.release()
        raise SystemExit

cv2.destroyWindow("SELECT LINE")

if len(points) < 5:

    cap.release()

    raise RuntimeError(
        "Select at least 5 points."
    )

p0 = np.asarray(
    points,
    dtype=np.float32
)

print()
print("Selected points:", len(points))

# ============================================================
# VERTICAL-ONLY TRACKING
# ============================================================

# For every selected x coordinate, search only vertically.
#
# This is the important change:
#
#       X = FIXED
#       Y = TRACKED
#
# We do NOT allow the tracker to move sideways.

search_radius = 15

reference_x = p0[:,0].astype(int)
reference_y = p0[:,1].astype(int)

# ============================================================
# TRACKING VIDEO
# ============================================================

overlay_path = OUT / "horizontal_line_VERTICAL_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(overlay_path),
    fourcc,
    fps,
    (W,H)
)

# Current y positions
current_y = reference_y.astype(float)

records = []

# Frame 0
for i in range(len(points)):

    records.append([
        0,
        i,
        reference_x[i],
        current_y[i],
        0.0
    ])

overlay = frame0.copy()

for i in range(len(points)):

    cv2.circle(
        overlay,
        (
            int(reference_x[i]),
            int(current_y[i])
        ),
        4,
        (0,255,0),
        -1
    )

writer.write(overlay)

# ============================================================
# FRAME LOOP
# ============================================================

for frame_idx in range(1,nframes):

    ok, frame = cap.read()

    if not ok:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    overlay = frame.copy()

    new_y = current_y.copy()

    for i in range(len(points)):

        x = reference_x[i]
        y0 = int(round(current_y[i]))

        y1 = max(
            2,
            y0 - search_radius
        )

        y2 = min(
            H - 3,
            y0 + search_radius
        )

        # ----------------------------------------------------
        # Small vertical strip around the point
        # ----------------------------------------------------

        patch = gray[
            y1:y2+1,
            max(0,x-2):min(W,x+3)
        ]

        if patch.size == 0:
            continue

        # Vertical brightness profile
        profile = np.mean(
            patch.astype(float),
            axis=1
        )

        # Smooth
        profile = cv2.GaussianBlur(
            profile.reshape(-1,1),
            (1,5),
            0
        ).ravel()

        # Vertical gradient
        grad = np.abs(
            np.gradient(profile)
        )

        # strongest local vertical edge
        k = int(
            np.argmax(grad)
        )

        candidate_y = y1 + k

        # Don't allow absurd movement
        if abs(candidate_y - y0) <= search_radius:

            new_y[i] = candidate_y

            records.append([
                frame_idx,
                i,
                x,
                candidate_y,
                candidate_y - reference_y[i]
            ])

            cv2.circle(
                overlay,
                (x, int(candidate_y)),
                4,
                (0,255,0),
                -1
            )

    current_y = new_y

    cv2.putText(
        overlay,
        "VERTICAL-ONLY LINE TRACK",
        (30,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0,255,0),
        2
    )

    writer.write(overlay)

cap.release()
writer.release()

records = np.asarray(
    records,
    dtype=float
)

tracking_file = (
    OUT /
    "horizontal_line_VERTICAL_tracking.npy"
)

np.save(
    tracking_file,
    records
)

print()
print("Tracking complete.")
print("Records:", len(records))
print("Saved:", tracking_file)

# ============================================================
# BUILD VERTICAL VIBRATION SIGNAL
# ============================================================

frame_id = records[:,0]
point_id = records[:,1]
x = records[:,2]
y = records[:,3]
dy = records[:,4]

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
# IMPORTANT:
# Use median across the selected X positions.
#
# They are all measuring the SAME vertical line motion.
# ------------------------------------------------------------

frames = np.unique(frame_id)

signal = []
signal_time = []
signal_y = []

for f in frames:

    mask = frame_id == f

    if np.sum(mask) < 3:
        continue

    vals = dy[mask]
    rows = y[mask]

    signal.append(
        np.median(vals)
    )

    # median physical row for rolling shutter
    signal_y.append(
        np.median(rows)
    )

    signal_time.append(
        f / fps
    )

signal = np.asarray(signal)
signal_y = np.asarray(signal_y)
signal_time = np.asarray(signal_time)

# ------------------------------------------------------------
# Rolling shutter timing
# ------------------------------------------------------------

times = (
    signal_time
    +
    signal_y * ROW_DELAY
)

order = np.argsort(times)

times = times[order]
signal = signal[order]

# ------------------------------------------------------------
# Audio interpolation
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
    signal
)

audio -= np.mean(audio)

# ------------------------------------------------------------
# Bass band
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

audio *= 2.0

audio = np.clip(
    audio,
    -1,
    1
)

wav = np.int16(
    audio * 32767
)

wav_path = (
    OUT /
    "horizontal_line_VERTICAL_bass.wav"
)

wavfile.write(
    wav_path,
    FS,
    wav
)

print()
print("==========================================")
print("VERTICAL LINE AUDIO")
print("==========================================")
print("Duration:", len(wav)/FS)
print("WAV:", wav_path)
print()
print("This version tracks Y ONLY.")
