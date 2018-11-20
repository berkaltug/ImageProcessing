import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img=mpimg.imread('some_img.jpg')
%matplotlib inline

def my_f1(image_1):
h_1={}
for i in range(image_1.shape[0]):
for j in range(image_1.shape[1]):
if(image_1[i,j,0]) in h_1.keys():
h_1[image_1[i,j,0]]=h_1[image_1[i,j,0]]+1
else:
h_1[image_1[i,j,0]]=1
return h_1
my_f1(img)
