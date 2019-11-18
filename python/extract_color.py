
import cv2
import numpy as np
from pprint import pprint



image_dir = '../image/'
image_file = 'half.png'
src_file  = image_dir + image_file

im = cv2.imread(src_file, 1) #(A) //ファイル読み込み

color = np.zeros(3)
for j in range(3):
    pprint (im[:,:,j])
    color[j] = np.median(im[:,:,j])

pprint(color)


