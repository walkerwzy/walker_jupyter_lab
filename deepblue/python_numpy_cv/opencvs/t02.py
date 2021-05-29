import cv2

image = cv2.imread("./cv2.png")

image_gary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("image_RGB", image)

cv2.imshow("image_gray", image_gary)

cv2.imshow("image_hsv", image_hsv)

if cv2.waitKey(0) == ord("q"):
    cv2.destroyAllWindows()

