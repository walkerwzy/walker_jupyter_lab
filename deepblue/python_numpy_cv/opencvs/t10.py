import cv2

img = cv2.imread("./cv2.png")

img[:,:,1] = 0

cv2.imshow("img", img)

cv2.waitKey(1000)

