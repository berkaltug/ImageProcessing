def convert_gray_level_to_BW(image_gray_level):
	im_BW=np.zeros((image_gray_level.shape[0],image_gray_level.shape[1]))
	for i in range(image_gray_level.shape[0]):
		for j in range(image_gray_level.shape[1]):
			if(image_gray_level[i,j]>120):
				im_BW[i,j]=0
			else:
				im_BW[i,j]=1
	return im_2
