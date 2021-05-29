import cv2

img = cv2.imread("./cv2.png")

eyes = img[77:89, 86:143]

# img[0:(89-77),0:(143-86)] = eyes

cv2.imshow("img", eyes)

cv2.waitKey(0)
