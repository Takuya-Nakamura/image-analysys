import cv2
import numpy as np
from pprint import pprint
from copy import deepcopy
import sys
import os



#入力画像をグレースケール変換＞２値化、二値化後の画像を返す
def getBinary(im):
    im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY) #(B) //グレースケールの取得
    return cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] #OTSUアルゴリズムを適用

# 外接矩形の取得
def getBoundingRectAngle(contour):
    # 配列構造が多重になっているので、１回シンプルな配列にする (point [x y])
    contour = list(map(lambda point: point[0], contour))
    x_list = [ point[0] for point in contour ] # width
    y_list = [ point[1] for point in contour ] # height
    return  [min(x_list), max(x_list), min(y_list), max(y_list)]

# 切り取り    
def getPartImageByRect(img, rect):
    #[y軸、x軸]
    return img[ rect[2]:rect[3], rect[0]:rect[1],]

# 輪郭データを y軸毎の最大、最小形式に変更(pointは0:x, 1:y)
# return [main_axis座標、target_axis座標左端、target_axis座標右端] 
# def getCropData(contour):
#     contour = list(map(lambda i: i[0], contour))
#     y_list = set([ point[1] for point in contour ]) # unique
#     arr = []
#     for y in y_list:
#         #輪郭配列から yが特定値のxの配列取得
#         points = list(filter(lambda i: i[1] == y, contour))
#         x_list = [ i[0] for i in points ]
#         arr.append( [ y, min(x_list), max(x_list)] )
#     return arr

def getCropData(contour, main_axis=0):
    target_axis =  1  if main_axis == 0 else 0

    #axis = 0 = x 
    #axis = 1 = y 
    contour = list(map(lambda i: i[0], contour))
    axis_point_list = set([ point[main_axis] for point in contour ]) # unique
    arr = []

    for val in axis_point_list:
        #輪郭配列から yが特定値のxの配列取得
        target_axis_point_list = list(filter(lambda i: i[main_axis] == val, contour))
        tmp_list = [ i[target_axis] for i in target_axis_point_list ] # 
        arr.append( [ val, min(tmp_list), max(tmp_list)] )
    return arr

def doCropY(input_im,  points, rect) :
    height = rect[3]-rect[2]
    width = rect[1]-rect[0] 
    left = rect[0]    
    top = rect[2]
    output_im = np.zeros((height, width, 4), np.uint8)

    for point in points :
        for x in range(0, width) :
            #input画像 座標
            in_y = point[0]
            in_x = x + left
            in_x_min = point[1]
            in_x_max = point[2]

            # output画像座標
            out_y = point[0] - top
            out_x = x
            out_x_min = point[1] - left
            out_x_max = point[2] - left

            #x軸の最大最小の範囲だったら元画像から新画像にコピーする
            if( out_x_min < x  and x < out_x_max):
                output_im[out_y : out_y + 1, out_x : out_x + 1, ] = input_im[in_y : in_y + 1, in_x : in_x + 1, ]

    return output_im


def doCropX(im, points, rect) :
    height = rect[3]-rect[2]
    width = rect[1]-rect[0] 
    left = rect[0]    
    top = rect[2]

    #pprint(points)

    for point in points :

        for y in range(0, height) :
            #input画像 座標
            y = y
            x = point[0] - left
            y_min = point[1] - top
            y_max = point[2] - top

            #pprint("###################")
            # 
            # pprint(x)
            # pprint("y:" + str(y))
            # pprint("y_min:" + str(y_min))
            # pprint("y_max:" + str(y_max))
            
            #y軸の最大最小の範囲だったら元画像から新画像にコピーする
            if(  y < y_min  or y_max < y):
                im[y:y+1, x:x+1,] = [0,0,0,0] #透過

    return im

##########################
#  main
##########################
outdir = './out/'
tempdir = './temporary/'

#引数から画像パス取得
args = sys.argv
if(len(args) <= 1):
    print("need arg 1. input file path.")
    sys.exit()
src_file = args[1]
root, extention = os.path.splitext(src_file)

# 画像読み込み
im_src = cv2.imread(src_file, -1) #アルファチャンネル込
# cv2.imwrite(tempdir + "original" + extention, im_src)

#binaryにして輪郭を取得
im_bin = getBinary(im_src)
#contours = cv2.findContours(im_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = cv2.findContours(im_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] # パスをすべて取得

for i, cnt in  enumerate(contours):
    #外接矩形の取得
    rect = getBoundingRectAngle(cnt)
    #y座標にxの左端、右端の範囲で切り取る
    crop_data  = getCropData(cnt, 1) #x軸基準
    im_out = doCropY(im_src, crop_data, rect)
    
    # #x座標毎にyの上から下の範囲外を透過させる
    crop_data  = getCropData(cnt, 0) #x軸基準
    im_out = doCropX(im_out, crop_data, rect)

    cv2.imwrite(outdir + "out" + str(i) + extention, im_out)
