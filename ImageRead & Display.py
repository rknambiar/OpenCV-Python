#This program uses opencv to read an image
#and display it using both matplotlib and opencv

#import statements
import cv2
import numpy as np
import matplotlib.pyplot as plt

#image read | Use your own image in place for lane.jpg
img = cv2.imread('lane.jpg',cv2.IMREAD_GRAYSCALE)

#image display
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#image display using matplotlib
plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.show()
