import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

VIDEO = r"C:\Users\Harkiran\Downloads\newlightvibeexp\MANUAL_LINE_tracking.mp4"
OUT = r"C:\Users\Harkiran\Downloads\newlightvibeexp"

X1, X2 = 700, 1050
Y1, Y2 = 750, 900

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

profiles = []

for k in range(n):

    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)

    # Horizontal average.
    # This is the actual physical image, NOT the green overlay.
    p = np.mean(gray[Y1:Y2, X1:X2], axis=1)

    profiles.append(p)

cap.release()

profiles = np.asarray(profiles)

print("profiles:", profiles.shape)

# ------------------------------------------------------------
# REMOVE EACH FRAME'S MEAN.
#
# This isolates the SHAPE across rows.
# ------------------------------------------------------------

shape = profiles - np.mean(profiles, axis=1, keepdims=True)

# ------------------------------------------------------------
# PLOT INDIVIDUAL FRAME PROFILES
# ------------------------------------------------------------

frames_to_show = [
    0,
    20,
    40,
    60,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    220
]

plt.figure(figsize=(14, 8))

for k in frames_to_show:
    if k < len(shape):
        plt.plot(
            np.arange(Y1, Y2),
            shape[k],
            label=f"frame {k}"
        )

plt.xlabel("Image Y / row")
plt.ylabel("Brightness relative to frame mean")
plt.title("INTRA-FRAME VIBRATION TEST")
plt.legend(ncol=3)

plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "INTRA_FRAME_profiles.png"),
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# SHOW CHANGE BETWEEN CONSECUTIVE ROWS
# ------------------------------------------------------------

row_difference = np.diff(shape, axis=1)

plt.figure(figsize=(14, 7))

plt.imshow(
    row_difference,
    aspect="auto",
    origin="upper",
    extent=[
        Y1,
        Y2 - 1,
        len(shape) / fps,
        0
    ]
)

plt.xlabel("Image Y / row")
plt.ylabel("Time (seconds)")
plt.title("INTRA-FRAME ROW DIFFERENCE")

plt.colorbar(label="Brightness difference")

plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "INTRA_FRAME_row_difference.png"),
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# HOW MUCH DOES EACH FRAME CHANGE ACROSS ROWS?
# ------------------------------------------------------------

intra_energy = np.std(shape, axis=1)

plt.figure(figsize=(14, 5))

plt.plot(
    np.arange(len(intra_energy)) / fps,
    intra_energy
)

plt.xlabel("Time (seconds)")
plt.ylabel("Within-frame row variation")
plt.title("INTRA-FRAME VIBRATION ENERGY")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "INTRA_FRAME_energy.png"),
    dpi=200
)

plt.close()

print()
print("Saved:")
print("INTRA_FRAME_profiles.png")
print("INTRA_FRAME_row_difference.png")
print("INTRA_FRAME_energy.png")