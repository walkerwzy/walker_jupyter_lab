import numpy as np
import cv2

image = np.zeros((500, 500, 3), dtype=np.uint8)


cv2.line(image, (0, 0), (499, 499), (0, 0, 255), 5)


cv2.rectangle(image, (249, 249), (499, 499), (0, 255, 0),  5)


cv2.circle(image, (249, 249), 20, (255, 0, 0), -1)



cv2.circle(image, (30, 30), 5, (255, 255, 255), -1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, "(30, 30)", (30, 30), font, 0.5,  (255, 255, 255), 1)

cv2.imshow("image", image)
cv2.waitKey(5000)


