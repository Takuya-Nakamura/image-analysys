import cv2
import os
import numpy as np
from pprint import pprint


def getRectByPoints(points):
    # prepare simple array 
    points = list(map(lambda x: x[0], points))

    points = sorted(points, key=lambda x:x[1])
    top_points = sorted(points[:2], key=lambda x:x[0])
    bottom_points = sorted(points[2:4], key=lambda x:x[0])
    points = top_points + bottom_points

    left = min(points[0][0], points[2][0])
    right = max(points[1][0], points[3][0])
    top = min(points[0][1], points[1][1])
    bottom = max(points[2][1], points[3][1])
    return (top, bottom, left, right)

def getPartImageByRect(rect, file):
    img = cv2.imread(file, 1)
    return img[rect[0]:rect[1], rect[2]:rect[3]]

def getMode(nparr):
    count = np.bincount(nparr) #値ごとの出現回数を計算
    return np.argmax(count) # 最大を要素のあちあｗ


##########################
# main
##########################

outdir = './sample1_out/'

image_dir = '../image/'
image_file = 'shikaku.png'
src_file  = image_dir + image_file

im = cv2.imread(src_file, 1) #(A) //ファイル読み込み
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #(B) //グレースケールの取得
im = cv2.GaussianBlur(im, (11, 11), 0) #(C) //ガウシアンフィルター(調整用)
cv2.imwrite(outdir + "gray.png", im)


##########################
# 二値化
##########################

ret1, th1 = cv2.threshold(im, 240, 255, cv2.THRESH_BINARY_INV) #二値化
th2 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3) #adaptiveThresholdここ調整の余地あり..
th3 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

cv2.imwrite(outdir + "bin.png", th1)
cv2.imwrite(outdir + "bin2.png", th2)
cv2.imwrite(outdir + "bin3.png", th3)

im_all = th1 + th2 + th3
cv2.imwrite(outdir + "bin_all.png", im_all)

##########################
# 輪郭抽出
##########################
contours = cv2.findContours(im_all, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

#画像サイズの1/100を基準面積として取得
th_area = im.shape[0] * im.shape[1] / 100 
#contoursに対してfilter
contours_large = list(filter(lambda c:cv2.contourArea(c) > th_area, contours))

outputs = []
rects = []
approxes = []
for (i,cnt) in enumerate(contours_large):
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*arclen, True)
    if len(approx) < 4:
        continue
    approxes.append(approx)
    rect = getRectByPoints(approx)
    rects.append(rect)
    outputs.append(getPartImageByRect(rect, src_file))
    cv2.imwrite(outdir + 'output'+str(i)+'.png', getPartImageByRect(rect, src_file))


##########################
# 代表色の取得
##########################
# http://peaceandhilightandpython.hatenablog.com/entry/2016/01/03/151320
#outputs
t_colors = []
for (i,out) in enumerate(outputs):
    color = np.zeros(3)
    for j in range(3):
        color[j] = np.median(out[:,:,j]) #中央値
        #color[j] = getMode(out[:,:,j].ravel()) #最頻値
    t_colors.append(color)

t_colors = np.array(t_colors)
pprint(t_colors)



