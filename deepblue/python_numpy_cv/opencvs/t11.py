import cv2

img1 = cv2.imread("./pic1.png")

img2 = cv2.imread("./pic2.png")

size = (300, 300)

img1 = cv2.resize(img1, size)

img2 = cv2.resize(img2, size)

dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

cv2.imshow("dst", dst)

cv2.waitKey(3000)
