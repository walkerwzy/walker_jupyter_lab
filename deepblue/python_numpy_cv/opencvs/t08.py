import cv2

img = cv2.imread("./cv2.png")
px = img[100, 100]
print(px)
px1 = img[100, 100, 0]
print(px1)

img[100:, 100:] = [255, 255, 255]

cv2.imshow("img", img)
cv2.waitKey(5000)

