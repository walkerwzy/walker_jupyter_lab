import numpy as np
import cv2 as cv

# https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html

# 512*512的图片，每个像素颜色为(0,0,0)
image = np.zeros((512, 512, 3), dtype='uint8')

cv.line(image, (0, 0), (511, 511), (255, 255, 255), 5)
cv.rectangle(image,(384,0),(510,128),(0,255,0),3)
cv.circle(image,(447,63), 63, (0, 0, 255), -1) # 将线宽为-1传给封闭图形时，将填充
cv.ellipse(image,(256, 256), (100,50),0,0,180,255,-1)

# polygon
pts = np.array([[[1,5],[20,30],[70,20],[50,10]]],np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(image,[pts],True,(0,255,255))

# add text
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(image, 'Hello World', (10, 500), font, 2, (255,255,255), 2, cv.LINE_AA)

cv.imshow('my draw', image)

if cv.waitKey(-1) == ord("q"):
    cv.destroyAllWindows()
