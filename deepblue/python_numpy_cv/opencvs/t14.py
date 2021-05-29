import cv2
import numpy as np

image = cv2.imread("./frog.png")
image_logo = cv2.imread("./frog_logo.png")

image_logo = cv2.resize(image_logo, (70, 70))
row, col, channel  = image_logo.shape
roi = image[0:row, 0:col, 0:channel]

image_logo_gray = cv2.cvtColor(image_logo, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(image_logo_gray, 245, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)

image_bg = cv2.bitwise_and(roi, image_logo, mask=mask)
image_logo_bg = cv2.bitwise_and(image_logo, image_logo, mask=mask_inv)

dst = cv2.add(image_bg, image_logo_bg)
image[0:row, 0:col] = dst

cv2.imshow("image", image)
cv2.waitKey(0)
