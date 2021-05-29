import cv2

image = cv2.imread("./cv2.png")  # RGB

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("cbq_hsv", image_hsv)

cv2.imwrite("./cbq_hsv.png", image_hsv)
#cv2.imshow("cbq_gray", image_gray)

# cv2.imshow("cbq", image)
k = cv2.waitKey(0)
if k == ord("q"):
    cv2.destroyAllWindows()
