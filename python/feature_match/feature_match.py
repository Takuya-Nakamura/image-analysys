import sys
import cv2
import numpy as np
import math
from pprint import pprint
#from IPython.display import display, Image
 
indir   = "./in/"
outdir  = "./out/"
tempdir = "./temporary/"

def main():
    target = cv2.imread(indir + "target1.jpeg")  # 検出元
    template = cv2.imread(indir + "template.jpeg") # template

    # A-KAZE検出器の生成
    detector = cv2.AKAZE_create()

    # 特徴量の検出と特徴量ベクトルの計算
    kp1, des1 = detector.detectAndCompute(target, None)
    kp2, des2 = detector.detectAndCompute(template, None)

    # Brute-Force Matcherの生成
    bf = cv2.BFMatcher()

    # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    # 第3引数で取得したk=2の数だけmatche候補が帰ってくる
    matches = bf.knnMatch(des1, des2, k=2)

    ##  [<DMatch 0x128937990>, <DMatch 0x1289379b0>],
    ### matchの中身
        # distance (特徴点の距離..類似度.小さい方が似ている)
        # imageIdx #トレーニング画像、テンプレート　
        # queryIdx # 探査対象の画像
        # trainIdx #トレーニング画像、テンプレート
    
    # 距離・類似度が低いものは省く。
    ratio = 0.2
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m]) 

    # 特徴量をマッチング状況に応じてソート
    good = sorted(matches, key = lambda x : x[1].distance) #distanceでソート 

    # 対応する特徴点同士を描画
    im_out = cv2.drawMatchesKnn(target, kp1, template, kp2, good[:2], None, flags=2)
    cv2.imwrite(outdir + "out.jpeg", im_out)




    ##### 切り出し
    q_kp = [] #対象ファイル
    t_kp = [] #トレーニングファイル

    for p in good[:2] :        
        for px in p:
            q_kp.append(kp1[px.queryIdx])
            t_kp.append(kp2[px.trainIdx])

    pprint("debug")
    pprint(q_kp[0].pt)
    pprint(q_kp[0].size)
    pprint(q_kp[0].angle)
    pprint(q_kp[0].response)

    ## 加工対象の画像から特徴点間の角度と距離を計算
    q_x1, q_y1 = q_kp[0].pt
    q_x2, q_y2 = q_kp[-1].pt #要素の最後

    q_deg = math.atan2(q_y2 - q_y1, q_x2 - q_x1) * 180 / math.pi #角度
    q_len = math.sqrt((q_x2 - q_x1) ** 2 + (q_y2 - q_y1) ** 2) #距離


    ## テンプレート画像から特徴点間の角度と距離を計算
    t_x1, t_y1 = t_kp[0].pt 
    t_x2, t_y2 = t_kp[-1].pt

    t_deg = math.atan2(t_y2 - t_y1, t_x2 - t_x1) * 180 / math.pi #角度
    t_len = math.sqrt((t_x2 - t_x1) ** 2 + (t_y2 - t_y1) ** 2) #距離

    # 切出し位置の計算
    x1 = int(q_x1 - t_x1 * (q_len / t_len))
    x2 = int(x1 + template.shape[1] * (q_len / t_len))

    y1 = int(q_y1 - t_y1 * (q_len / t_len))
    y2 = int(y1 + template.shape[0] * (q_len / t_len))

    pprint(x1)
    pprint(x2)
    pprint(y1)
    pprint(y2)

    # 画像サイズ
    x, y, c = target.shape
    size = (x, y)



    # 回転の中心位置
    center = (q_x1, q_y1)

    # 回転角度
    angle = q_deg - t_deg

    # サイズ比率
    scale = 1.0

    # 回転変換行列の算出
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # アフィン変換(rotate)
    img_rot = cv2.warpAffine(target, rotation_matrix, size, flags=cv2.INTER_CUBIC)

    # 画像の切出し
    img_rot = img_rot[y1:y2, x1:x2]

    # # 縮尺調整
    x, y, c = template.shape
    img_rot = cv2.resize(img_rot, (y, x))

    # 結果表示
    #display_cv_image(img_rot, '.png')
    cv2.imwrite(outdir + "out2.jpeg", img_rot)

######################
# functions
######################

    


#### main ####
main()