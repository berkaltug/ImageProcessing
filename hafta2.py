import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img=mpimg.imread('test_1.jpg')
%matplotlib inline
plt.imshow(img)

def my_function_1(my_img):
print("eksen sayısı: ", my_img.ndim)
print("eksen degeri: ", my_img.shape)
print("en küçük kırmızı renk degeri: ",np.min(my_img[:,:,0]))
print("en büyük kırmızı renk degeri: ",np.max(my_img[:,:,0]))

print("en küçük kırmızı renk degeri: ",np.min(my_img[:,:,1]))
print("en büyük kırmızı renk degeri: ",np.max(my_img[:,:,1]))

print("en küçük kırmızı renk degeri: ",np.min(my_img[:,:,2]))
print("en büyük kırmızı renk degeri: ",np.max(my_img[:,:,2]))
my_function_1(img)
