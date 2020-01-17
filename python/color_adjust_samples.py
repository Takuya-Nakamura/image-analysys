import cv2
import numpy as np
from pprint import pprint
from copy import deepcopy
import sys
import os

def getColorMedian(im):
    color = [0, 0, 0]
    for j in range(3):
        color[j] = np.median(im[:,:,j]) 
    return color


##########################
# <処理の流れ>
# 輪郭抽出と切り取り手順の整理
##########################
outdir = './out/'
tempdir = './temporary/'

rgb_base = [255,255,255]
hsv_base = [0, 0, 255]

#引数取得
args = sys.argv
if(len(args) <= 1):
    print("need arg 1. input file path.")
    sys.exit()

src_file = args[1]
root, extention = os.path.splitext(src_file)

# 画像取得
im_src_rgb = cv2.imread(src_file, 1) #(A) ファイル読み込み
im_src_hsv = cv2.cvtColor(im_src_rgb, cv2.COLOR_BGR2HSV) #hsvに変換

pprint(im_src_hsv)
# im_bgr = cv2.cvtColor(im_src_hsv, cv2.COLOR_HSV2BGR) # bgrに変換して出力
# cv2.imwrite(tempdir + "meido" +str(val)+ extention, im_bgr)

hsv_input_median = getColorMedian(im_src_hsv[0:5, 0:5])

pprint(hsv_input_median)
#明度の調整
im_src_hsv[:, :, [2]] = im_src_hsv[:,:, [2]] * (hsv_base[2] / hsv_input_median[2])

pprint(im_src_hsv[:,:])