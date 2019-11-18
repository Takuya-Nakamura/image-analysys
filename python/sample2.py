

import cv2
import numpy as np
from pprint import pprint
from copy import deepcopy

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

#入力画像をグレースケール変換＞２値化、二値化後の画像を返す
def getBinary(im, filename):
    im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY) #(B) //グレースケールの取得
    im_bin = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] #OTSUアルゴリズムを適用
    cv2.imwrite(outdir + filename + "_gray" + extention, im_gray)
    cv2.imwrite(outdir + filename + "_bin" + extention, im_bin)

    return im_bin

##########################
# <処理の流れ>
# 元画像から　２値画像を作成する
# ２値画像から輪郭を抽出する
# 輪郭情報から四角で切り抜く
# 各切り抜いた部分の代表色を取得する
##########################
outdir = './sample2_out/'

image_dir = '../image/'
image_file = 'whitebord'
extention = ".jpeg"
src_file  = image_dir + image_file + extention

im_src = cv2.imread(src_file, 1) #(A) //ファイル読み込み


##########################
# 二値化
##########################
bin_images = []
im = deepcopy(im_src)
bin_images.append(getBinary(im, "original"))

#それぞれの色味だけ(青)
im = deepcopy(im_src)
im[:,:,[1,2]] = 0 #g rを0
bin_images.append(getBinary(im, "blue"))

#それぞれの色味だけ(緑)
im = deepcopy(im_src)
im[:,:,[0,2]] = 0 #b rを0
bin_images.append(getBinary(im, "green"))

#それぞれの色味だけ(赤)
im = deepcopy(im_src)
im[:,:,[0,1]] = 0 #b gを0
bin_images.append(getBinary(im, "red"))


##  赤POSTITの取得ができない。
# 以下でサンプルで記載された内容はやってみたが(3つのカラーチャネルの赤から他の青と緑を引いたものの和)をセット
# 帰って抽出できるポストイットが減ってしまった。ここらへんはチューニングに感覚というか、経験が必要そう。
# im = deepcopy(im_src)
# im[:,:,2] = (np.abs(im[:,:,2] - im[:,:,1]) + np.abs(im[:,:,2] - im[:,:,0]))
# #im[:,:,[0,1]] = 0
# bin_images.append(getBinary(im, "option"))

# binaryのマージをして最終的な輪郭検出用のbinrayを作成する
im_all = sum(bin_images)
cv2.imwrite(outdir + "bin_all" + extention, im_all)


##########################
# 輪郭抽出
##########################
contours = cv2.findContours(im_all, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

#debug:輪郭描画
contours_img = cv2.drawContours(im_src, contours, -1, (0,0,0), 5)
cv2.imwrite(outdir + "contours" + extention, contours_img)

## 切り抜き処理
#画像サイズの1/100を基準面積として取得
th_area = im.shape[0] * im.shape[1] / 100
#contoursに対して一定のサイズ以上のエリアにfilter
contours_large = list(filter(lambda c:cv2.contourArea(c) > th_area, contours))

# 切り取り処理
outputs = []
rects = []
approxes = []
for (i,cnt) in enumerate(contours_large):
    arclen = cv2.arcLength(cnt, True) # 輪郭線の周囲長さ、あるいは曲線の長さを返す
    approx = cv2.approxPolyDP(cnt, 0.02 * arclen, True) # 折れ線(カーブ)を指定された精度で近似する
    if len(approx) < 4:
        continue
    approxes.append(approx)
    rect = getRectByPoints(approx)
    rects.append(rect)
    outputs.append(getPartImageByRect(rect, src_file))
    cv2.imwrite(outdir + 'output' + str(i) + extention, getPartImageByRect(rect, src_file))


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
print("代表色各切り抜きの代表色")
pprint(t_colors)


