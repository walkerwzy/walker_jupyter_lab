import numpy as np
import cv2

def nothing(x):
    pass

image = np.zeros((300, 300, 3), np.uint8)
cv2.namedWindow("image")

cv2.createTrackbar("R", "image", 0, 255, nothing)
cv2.createTrackbar("G", "image", 0, 255, nothing)
cv2.createTrackbar("B", "image", 0, 255, nothing)

switch = "0:OFF / 1:ON"

cv2.createTrackbar(switch, "image", 0, 1, nothing)

while True:
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break

    r = cv2.getTrackbarPos("R", "image")
    g = cv2.getTrackbarPos("G", "image")
    b = cv2.getTrackbarPos("B", "image")
    s = cv2.getTrackbarPos(switch, "image")

    if s == 0:
        image[:] = 0
    else:
        image[:] = [b, g, r]
cv2.destroyAllWindows()

