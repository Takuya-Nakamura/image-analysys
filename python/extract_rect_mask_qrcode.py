import cv2
import numpy as np
from pprint import pprint
from copy import deepcopy
import sys
import os
import math
import shutil


def getBoundingRectAngle(contour):
    # 配列構造が多重になっているので、１回シンプルな配列にする (point [x y])
    contour = list(map(lambda point: point[0], contour))
    x_list = [ point[0] for point in contour ] # width
    y_list = [ point[1] for point in contour ] # height
    return  [min(x_list), max(x_list), min(y_list), max(y_list)]

# 切り取り    
def getPartImageByRect2(rect, im):
    #[y軸、x軸]
    return im[ rect[2]:rect[3], rect[0]:rect[1],]


def getColorMedian(im):
    color = [0, 0, 0]
    for j in range(3):
        color[j] = math.floor(np.median(im[:,:,j]))
    return color


def getRgbHex(bgr):
    hex_code = "#"
    rgb = reversed(bgr)
    for val in rgb :
        hex_code += hex(val)[2:]
    return hex_code

def getBinary(im, filename):
    im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY) #(B) //グレースケールの取得
    im_bin = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] #OTSUアルゴリズムを適用
    cv2.imwrite(tempdir + filename + "_gray" + extention, im_gray)
    cv2.imwrite(tempdir + filename + "_bin" + extention, im_bin)
    return im_bin

## 横が縦より２倍以上長い場合長方形と判断
def isRectAngle(im):
    height, width=im.shape[0:2]
    
    return  height * 2 < width 

## 横と縦の比率が10%以内なら正方形と判断
def isSuare(im):    
    x=im_src.shape[0]
    y=im_src.shape[1]
    return  (x / y)  < 1.1 


## getWidthFromPoinrt　(points x:y)
def getPointData(points):
    points = list(map(lambda point: point[0], points))
    pprint(points)
    left_top = points[0] 
    right_top = points[1] 
    right_bottom = points[2] 
    left_bottom = points[3]
    return left_top, left_bottom, right_bottom, right_top

def getAngle(point1_x,point1_y, point2_x, point2_y) :
    #2点間の傾きを取得
    radians = math.atan2(point1_y-point2_y, point2_x - point1_x)
    degrees = math.degrees(radians)
    return degrees

def rotate(im, angle):
    height = im.shape[0]                         
    width = im.shape[1]  
    center = (int(width/2), int(height/2))
    scale = 1.0
    trans = cv2.getRotationMatrix2D(center, angle , scale)
    return  cv2.warpAffine(im, trans, (width,height))


def adjustColorImage(left_top_x, left_top_y):
    size = 100 # 引き渡された位置から正方形に切り出す
    im_color_compare = im_src[
        left_top_y : left_top_y + size, # height
        left_top_x : left_top_x + size, # width 
    ]

    #debug
    cv2.imwrite(tempdir + 'color_compare'+ extention, im_color_compare)

    # 本来の背景色
    original_back_color = [214,214,214]
    input_back_color = getColorMedian(im_color_compare) # 代表色取得

    for i in range(3):
        ratio = (original_back_color[i]/input_back_color[i])
        im_src[:,:,i] = im_src[:,:,i] * ratio

    # 補正後
    cv2.imwrite(tempdir + "adjusted" + extention, im_src)
    return im_src


##########################
# <処理の流れ>
##########################
########################################
## チップの台の抽出処理
## QRコードを検出し、QRコードの位置からチップ台の位置を計算する。
## そのそれぞれの画像に２値化をかけてチップを取得する。
## 
########################################

# 初期処理
outdir = './out/'
tempdir = './temporary/'

shutil.rmtree(outdir)
os.mkdir(outdir)

shutil.rmtree(tempdir)
os.mkdir(tempdir)


#引数取得
args = sys.argv
if(len(args) <= 1):
    print("need arg 1. input file path.")
    sys.exit()

src_file = args[1]
root, extention = os.path.splitext(src_file)
im_src = cv2.imread(src_file, 1) #(A) //ファイル読み込み

#qrコード検出
detector = cv2.QRCodeDetector()
data, points, straight_qrcode = detector.detectAndDecode(im_src)
if data:
    print(f'decoded data: {data}')
    for i in range(4):
        cv2.line(im_src, tuple(points[i][0]), tuple(points[(i + 1) % len(points)][0]), (0, 0, 255), 5)
    cv2.imwrite(tempdir + 'qrcode' + extention, im_src)
else:
  print("no qrcode detected. exit.")  
  sys.exit()  

left_top, left_bottom, right_bottom, right_top = getPointData(points)

#傾き補正
angle = getAngle(left_top[0], left_top[1], right_top[0],right_top[1])
im_src = rotate(im_src, -angle) #傾きを戻したいので、angleに-をかける
# cv2.imwrite(tempdir + 'rotated' + extention, im_src)



########################################
## QRコード位置から各要素の角位置を計算する
########################################
## 幅・高さ・中央の計算
qr_width = right_bottom[0] - left_bottom[0] 
qr_height = left_bottom[1] - left_top[1]
qr_middle_x = int(left_top[0] + qr_width/2)

## qrコードは640 * 640
# チップ台までおおよそ920   920/640 = 1.4375
frmo_qr_bottom_to_chip_base = qr_height * 1.43

# チップ台は  1200 * 270 (大きめにとったほうが安全)
##  1200/640  = 1.875 
##  270/640 = 0.421875
chip_base_width = qr_width *  1.95
chip_base_height = qr_height * 0.55
qr_bottom_left_y = left_bottom[1]

### 色補正エリアを取得してイメージ調整
left_top = (
    qr_middle_x,  #x座標 qrコードの中心
    int(qr_bottom_left_y + frmo_qr_bottom_to_chip_base / 4) # y座標 qrコードとチップ台紙の中間
)
im_src = adjustColorImage(left_top[0], left_top[1])


#### chip台の位置 x,y 
chip_left_top =  (
    int(qr_middle_x - chip_base_width/2),
    int(qr_bottom_left_y + frmo_qr_bottom_to_chip_base),
)

chip_right_top = (
    int(qr_middle_x + chip_base_width/2 ),
    chip_left_top[1], # y
)

chip_right_bottom = (
    chip_right_top[0],
    int(chip_left_top[1] + chip_base_height ), # y
)

chip_left_bottom = (
    chip_left_top[0],
    chip_right_bottom[1],# y
)


## チップ台切り抜き[y:y,x:x ]
im_chipbase = im_src[ chip_left_top[1]:chip_left_bottom[1], chip_left_top[0]: chip_right_top[0] ]
cv2.imwrite(tempdir + 'chipbase'+ extention, im_chipbase)

im_debug = deepcopy(im_src)
cv2.rectangle(im_debug, chip_left_top, chip_right_bottom, (0, 0, 255), 5)
cv2.imwrite(tempdir + 'chipbase_counstrous'+ extention, im_debug)


########################################
## チップ台からチップを抽出する処理
## 切り取ったチップ台の中心を基準にセンターのチップの
## 左端と右端の位置を決めて３つのチップ画像に分割する 
########################################
# チップ自体は 250 250
# 250/640 = 0.390625
 
def split_chip_base(im_chipbase, qr_width):
    # 中心を取得
    height, width=im_chipbase.shape[0:2]
    width_middle  = width / 2
    
    # chipサイズ中央から左右に位置を取得
    left_separator_point = int(width_middle - (qr_width * 0.4) / 1.5)
    right_separator_point = int(width_middle + (qr_width * 0.4) / 1.5 )

    left  = im_chipbase[:,  0:left_separator_point,]
    center = im_chipbase[:,left_separator_point:right_separator_point]
    right = im_chipbase[:, right_separator_point:]

    return  [left, center, right ]



#入力画像をグレースケール変換＞２値化、二値化後の画像を返す
def mask_by_binary(im, name):
    im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY) #(B) //グレースケールの取得
    im_bin = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] #OTSUアルゴリズムを適用

    return crop_contours(im, im_bin, name)

#### イメージの輪郭抽し切り取り(大きなサイズ一つ)
def crop_contours(im_org, im_mask, name):
    contours = cv2.findContours(im_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    ## 切り抜き処理
    area_min = im_org.shape[0] * im_org.shape[1] / 3
    area_max = im_org.shape[0] * im_org.shape[1] * 0.9
     
    chip = "" 
    for (i,cnt) in enumerate(contours):
        #size fileter
        if (cv2.contourArea(cnt) <  area_min or area_max < cv2.contourArea(cnt)):
            continue
        # 四角に限定
        arclen = cv2.arcLength(cnt, True) # 輪郭線の周囲長さ、あるいは曲線の長さを返す
        approx = cv2.approxPolyDP(cnt, 0.02 * arclen, True) # 折れ線(カーブ)を指定された精度で近似する
        if len(approx) < 4:
            continue
            
        rect = getBoundingRectAngle(approx)
        chip = getPartImageByRect2(rect, im_org)
        cv2.imwrite(outdir + 'chips_' + name  + extention, chip)
        break

    return chip

def getChip(im, name) :
    im_masked = deepcopy(im)
    # binaryマスク作成
    im_masked = mask_by_binary(im_masked, name)
    return im_masked

# 代表色の取得
def getMainColorRgb(im):
    color = np.zeros(3)
    for j in range(3):
        color[j] = np.median(im[:,:,j]) #中央値
    return [ int(color[2]), int(color[1]), int(color[0])]

def getRgbHex(rgb):
    pprint(rgb)
    hex_code = "#"
    #rgb = reversed(bgr)
    for val in rgb :
        hex_code += hex(val)[2:]
    return hex_code


labels = ["left", "center","right"]
chip_bases = split_chip_base(im_chipbase, qr_width)

colors = {}
colors_hex = {}
for i in range(3):
    im_chip = getChip(chip_bases[i], labels[i])
    colors[labels[i]] = getMainColorRgb(im_chip)
    colors_hex[labels[i]] = getRgbHex(getMainColorRgb(im_chip))

pprint(colors)
pprint(colors_hex)








