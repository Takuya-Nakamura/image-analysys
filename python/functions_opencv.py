import cv2
import numpy as np
from pprint import pprint
from copy import deepcopy
import sys
import os
import math
import shutil

def getBoundingRectAngle(contour):
    """
    輪郭から外接矩形を取得する

    Args:
        contour contour:
            OpenCVの輪郭情報
    Returns:
        arr[left, right, top, bottom ]:
            輪郭の4つの角
    """
    # 配列構造が多重になっているので、１回シンプルな配列にする (point [x y])
    contour = list(map(lambda point: point[0], contour))
    x_list = [ point[0] for point in contour ] # width
    y_list = [ point[1] for point in contour ] # height
    return  [min(x_list), max(x_list), min(y_list), max(y_list)]

def getPartImageByRect(rect, im):
    """
    四角情報をimから切り取る

    Args:
        rect arr[left, right, top, bottom ]:
            OpenCVの輪郭情報
    Returns:
        im: 抽出後のイメージデータ
    """

    #[y軸、x軸]
    return im[ rect[2]:rect[3], rect[0]:rect[1],]


def getColorMedian(im):
    """
    画像の色の代表値を取得する

    Args:
        im image:
            抽出対象のイメージファイル
    Returns:
        arr :
        bgr色配列
    """
    color = [0, 0, 0]
    for j in range(3):
        color[j] = math.floor(np.median(im[:,:,j]))
    return color


def getRgbHex(bgr):
    """
    bgrカラーコードを16進(#xxxxxx)形式に変換する

    Args:
        bgr arr[int, int, int]:
            bgrコード
    Returns:
        str :
        "#xxxxxx"
    """
    hex_code = "#"
    rgb = reversed(bgr)
    for val in rgb :
        hex_code += hex(val)[2:]
    return hex_code


def isRectAngle(im):
    """
    imgが長方形か判定

    Args:
        im im:
            ファイルデータ
    Returns:
        boolean :
            長方形ならtrue
    """

    height, width=im.shape[0:2]
    
    return  height * 2 < width 

def isSquare(im):    
    """
    imgが正方形か判定

    Args:
        im im:
            ファイルデータ
    Returns:
        boolean :
            正方形ならtrue
    """

    x=im_src.shape[0]
    y=im_src.shape[1]
    return  (x / y)  < 1.1 


def getPointData(points):
    """
    QRコード検出の戻り値pointsから4角のpoint情報を取得する
    戻り値のpointsのフォーマットが配列が多かったり、分かりづらいので
    整形する目的

    Args:
        points arr[arr[point]]:
            角情報
    Returns:
        boolean :
            正方形ならtrue
    """

    points = list(map(lambda point: point[0], points))
    pprint(points)
    left_top = points[0] 
    right_top = points[1] 
    right_bottom = points[2] 
    left_bottom = points[3]
    return left_top, left_bottom, right_bottom, right_top

def getAngle(point1_x, point1_y, point2_x, point2_y) :
    """
    2点間の傾きを取得する

    Args:
        2点の(x,y)
    Returns:
        int :
            角度
    """

    #2点間の傾きを取得
    radians = math.atan2(point1_y-point2_y, point2_x - point1_x)
    degrees = math.degrees(radians)
    return degrees

def rotate(im, angle):
    """
    画像全体を傾ける

    Args:
        im:
            対象画像データ
        angle:
            傾ける角度    
    Returns:
        im :
            傾けた画像データ
    """

    height = im.shape[0]                         
    width = im.shape[1]  
    center = (int(width/2), int(height/2))
    scale = 1.0
    trans = cv2.getRotationMatrix2D(center, angle , scale)
    return  cv2.warpAffine(im, trans, (width,height))










def mask_by_binary(im, name):
    """
    imgをグレースケールに変換し、二値化して色の強い部分だけを切り出す。

    Args:
        im im:
            ファイルデータ
        name str:
            出力ファイル名    
    Returns:
        im :
        抽出簿の画像データ
    """
    im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY) #(B) //グレースケールの取得
    im_bin = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] #OTSUアルゴリズムを適用

    return crop_contours(im, im_bin, name)

def crop_contours(im_org, im_mask, name):
    """
    mask画像から輪郭を取得し、その輪郭に沿って元画像から切り出す。
    サイズと形でフィルターし、一つ見つかったら終了としている。
    Args:
        im_org im:
            元画像

        im_mask im:
            輪郭を抽出する画像(2値化済み、mask画像など)
         nam str:
            出力ファイル名   
    
    Returns:
        im :
        抽出簿の画像データ
    """

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
        chip = getPartImageByRect(rect, im_org)
        cv2.imwrite(outdir + 'chips_' + name  + extention, chip)
        break

    return chip




