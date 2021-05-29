import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("../img/frog2.jpg")
img2 = cv.imread("../img/frog1.jpg")
# img1 = cv.resize(img1, (330, 330))
img2 = cv.resize(img2, (90, 90))

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols, 0:channels]

img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 主体是黑色，背景是白色(1)，显然可以用来擦除主体(背景用)
ret, mask_bg = cv.threshold(img2gray, 190, 255, cv.THRESH_BINARY)
# 反相用来擦除背景（前景抠图用）
mask_fg = cv.bitwise_not(mask_bg)

bg = cv.bitwise_and(roi, roi, mask=mask_bg)
fg = cv.bitwise_and(img2, img2, mask=mask_fg)

img = cv.add(bg, fg)
img1[0:rows, 0:cols] = img

# cv.imshow("senior", img1)
# cv.waitKey(0)

fig, ax = plt.subplots(nrows=2, ncols=3)
fig.subplots_adjust(hspace=.5)
x1, x2, x3, x4, x5, x6 = ax.flatten()
x1.set_title('back ground')
x2.set_title('fore ground')
x3.set_title('mix fore')
x4.set_title('mix roi')
x5.set_title('mixed')
x6.set_title('full')
x1.imshow(mask_bg)
x2.imshow(mask_fg)
x3.imshow(fg)
x4.imshow(bg)
x5.imshow(img)
x6.imshow(img1)
plt.show()