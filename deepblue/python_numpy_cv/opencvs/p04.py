import cv2
import numpy as np

def draw(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(image, (x, y), 1, (225, 0, 0), 1)

    image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.setMouseCallback("image", draw)

    while True:
        cv2.imshow("image", image)
        if cv2.waitKey(0) == ord("q"):
            break
    cv2.destroyWindows()

