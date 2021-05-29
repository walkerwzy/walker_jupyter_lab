import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

image = np.zeros((500, 500, 3), np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_circle)
while True:
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
