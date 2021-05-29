import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img, (x, y), 15, (0, 0, 255), -1)
        cv2.imshow("image", img)

img = np.zeros((512, 512, 3), np.int8)
cv2.namedWindow('image')
cv2.setMouseCallback("image", draw_circle)

while True:
    cv2.imshow("image", img)
    if cv2.waitKey(0) == ord("q"):
        break
cv2.destroyAllWindows()
