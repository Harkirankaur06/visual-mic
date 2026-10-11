import cv2
import numpy as np
import os

VIDEO = r"C:\Users\Harkiran\Downloads\Mobile Devices\new vid.mp4"
OUTDIR = r"C:\Users\Harkiran\Downloads\light vibe exp"

os.makedirs(OUTDIR, exist_ok=True)

# Tight region around YOUR U-shaped water edge
x1, x2 = 160, 340
y1, y2 = 760, 940

cap = cv2.VideoCapture(VIDEO)

ok, first = cap.read()
if not ok:
    raise RuntimeError("Could not read video")

first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

roi = first_gray[y1:y2, x1:x2]

# Slight smoothing
roi_blur = cv2.GaussianBlur(roi, (5,5), 0)

# Find trackable feature points
points = cv2.goodFeaturesToTrack(
    roi_blur,
    maxCorners=40,
    qualityLevel=0.03,
    minDistance=5,
    blockSize=7
)

if points is None:
    raise RuntimeError("No trackable points found")

# Convert ROI coordinates -> full-frame coordinates
points[:,0,0] += x1
points[:,0,1] += y1

print("Initial points:", len(points))

prev_gray = first_gray
tracks = [points.reshape(-1,2).copy()]

lk_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)

while True:

    ok, frame = cap.read()

    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        points,
        None,
        **lk_params
    )

    if new_points is None:
        break

    good_new = new_points[status == 1]
    good_old = points[status == 1]

    # Keep only reasonable movements
    displacement = np.linalg.norm(
        good_new - good_old,
        axis=1
    )

    valid = displacement < 8.0

    good_new = good_new[valid]

    # Keep same number of points only if possible
    if len(good_new) < 3:
        break

    tracks.append(good_new.copy())

    # Reuse tracked points
    points = good_new.reshape(-1,1,2)
    prev_gray = gray

cap.release()

# ---------------------------------------------------------
# Save trajectories
# ---------------------------------------------------------

min_points = min(len(t) for t in tracks)

T = np.array([
    t[:min_points]
    for t in tracks
])

np.savez_compressed(
    os.path.join(
        OUTDIR,
        "water_edge_feature_tracks.npz"
    ),
    tracks=T,
    fps=30.0,
    roi=np.array([x1,x2,y1,y2])
)

# ---------------------------------------------------------
# Diagnostic image
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 140)

ok, frame = cap.read()
cap.release()

if ok:

    vis = frame.copy()

    # Draw initial/current tracked positions
    for p in T[140]:

        px = int(p[0])
        py = int(p[1])

        cv2.circle(
            vis,
            (px,py),
            4,
            (0,0,255),
            -1
        )

    # Draw trajectories
    for j in range(min_points):

        pts = T[:,j,:].astype(np.int32)

        for k in range(1,len(pts)):

            cv2.line(
                vis,
                tuple(pts[k-1]),
                tuple(pts[k]),
                (0,255,0),
                1
            )

    # ROI rectangle
    cv2.rectangle(
        vis,
        (x1,y1),
        (x2,y2),
        (255,0,0),
        2
    )

    cv2.imwrite(
        os.path.join(
            OUTDIR,
            "water_edge_feature_tracking.png"
        ),
        vis
    )

print()
print("Frames tracked:", len(T))
print("Tracked points:", min_points)

print()
print("Saved:")
print(
    os.path.join(
        OUTDIR,
        "water_edge_feature_tracking.png"
    )
)
print(
    os.path.join(
        OUTDIR,
        "water_edge_feature_tracks.npz"
    )
)
