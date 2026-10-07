import cv2
import time
import numpy as np

WINDOW = "10 Hz Calibration Target"

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

period = 1.0 / 200.0
half_period = period / 2.0

start = time.perf_counter()

while True:

    elapsed = time.perf_counter() - start

    state = int(elapsed / half_period) % 2

    if state == 0:
        frame = 255 * np.ones((1080, 1920), dtype=np.uint8)
    else:
        frame = np.zeros((1080, 1920), dtype=np.uint8)

    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cv2.destroyAllWindows()