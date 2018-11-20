import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img=mpimg.imread('image.jpg')%matplotlib inline

def getmean(image):
x=image.shape[0]
y=image.shape[1]
z=x*y
toplam=0
for i in range(x):
for j in range(y):
toplam=toplam+(image[i,j,0]+image[i,j,1]+image[i,j,2])/3
toplam=toplam/z
return toplam
getmean(img)
