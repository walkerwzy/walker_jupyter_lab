import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = np.ones((50,50,3), np.uint8)
img2 = np.ones((50,50,3), np.uint8)
img1[:,:,1:2] = 100
img2[6:14,6:14,:] = 255

img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

ret, bg_mask = cv.threshold(img2gray, 190, 255, cv.THRESH_BINARY)
fg_mask = cv.bitwise_not(bg_mask)



plt.imshow(fg_mask)
plt.show()
plt.imshow(bg_mask)
plt.show()
plt.imshow(img2gray)
plt.show()
plt.imshow(cv.bitwise_and(img1, img2))
