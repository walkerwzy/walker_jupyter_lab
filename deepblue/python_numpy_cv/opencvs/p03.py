import cv2
import numpy as np

image = np.zeros((300, 300, 3), dtype=np.uint8)

cv2.line(image, (0, 0), (149, 299), (255, 0, 255), 5)
cv2.line(image, (0, 0), (299, 149), (0, 0, 255), 5)

cv2.rectangle(image, (0, 0), (149, 149), (255, 255, 255), 1)

cv2.circle(image, (149, 149), 149, (255, 255, 0), -1)

cv2.putText(image, "(30, 30)", (30, 30), 4, (255, 255, 255), 5 )

cv2.imshow("image", image)
cv2.waitKey(0)

