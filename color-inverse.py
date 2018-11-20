import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img=mpimg.imread('image.jpg')
%matplotlib inline

plt.subplot(1,2,1),plt.imshow(img)
plt.subplot(1,2,2),plt.imshow(255-img)
