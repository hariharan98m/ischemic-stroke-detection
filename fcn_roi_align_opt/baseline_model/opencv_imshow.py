import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('fig_in_question.png')

# im.shape
#
# np.argwhere(im == [255,0,0])
#
# plt.imshow(im)
#
# im.shape

img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# plt.imshow(img_gray, cmap='gray')

cv2.imshow('Gray image', img_gray)
cv2.waitKey(0)
# cv2.destroyAllWindows()
