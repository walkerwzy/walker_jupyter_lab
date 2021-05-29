import cv2

image = cv2.imread("./cv2.png")

image[2] = 0  #  B G R [3]

cv2.imshow("img", image)
cv2.waitKey(5000)
