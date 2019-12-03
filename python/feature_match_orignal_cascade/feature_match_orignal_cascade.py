import sys
import cv2
import numpy as np
import math
from pprint import pprint
#from IPython.display import display, Image
 
indir   = "./in/"
outdir  = "./out/"
tempdir = "/temporary/"

def main():
    '''
    画像から特定の物体を検出したい。全く同じものではなくて、類似したもの,,,
    魚とか..顔検出はもともとあるみたい。

    '''
    # 参考:https://www.pro-s.co.jp/blog/system/opencv/6202

    # 入力画像の読み込み（テスト用画像ファイル）
    img = cv2.imread("./pos/spoon3.jpg")
    
    # カスケード型識別器（自作した分類器）
    cascade = cv2.CascadeClassifier("./cascade1/cascade.xml") # 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # face→ballに変更（そのままでもいいですけど）
    ball = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(0, 0))

    # 顔領域を赤色の矩形で囲む
    for (x, y, w, h) in ball:
        pprint(w * h )
        cv2.rectangle(img, (x, y), (x + w, y+h), (0,0,200), 3)

    # 結果画像を保存
    cv2.imwrite("result.png",img)
    



#### main ####
main()