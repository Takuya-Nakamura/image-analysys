# Cvmatのgemリファレンス
# https://www.rubydoc.info/gems/ruby-opencv/OpenCV/CvMat#resize-instance_method
# https://www.rubydoc.info/gems/ruby-opencv/OpenCV/CvSize

#ruby参考
# https://marunouchi-tech.i-studio.co.jp/2172/


require 'opencv'
include OpenCV

require 'pp'

image_folder = '../image/'
image_file  = 'shikaku.png'
src_img = CvMat.load(image_folder + image_file)  #matrixオブジェクトにロード

# ここが白以外の色部分を抽出する所。抽出についてはここ、を細かく調整する様子。
# gray_scaleを複数パターンでやって和をとって、とかそういうことができるみたい。
# thresholdの第３引数とか、閾値の範囲も調整が非常に難しい。
gray_img = src_img.BGR2GRAY #グレースケール
bin_img = gray_img.threshold(240, 255, :binary) #2値化(127-255の範囲が白)
bin_img = bin_img.not #後の輪郭抽出のために白黒反転

# 表示
# GUI::Window.new('gray').show(gray_img)
# GUI::Window.new('bin').show(bin_img)
# GUI::wait_key

# 画像保存
gray_img.save_image("gray.png")
bin_img.save_image("bin.png")

##### 輪郭の座標を取得 #####
contours = bin_img.find_contours({
    mode:1, #default 1
    method:1 #default 2
})


black = CvScalar.new(0,0,0)
blue = CvScalar.new(255,0,0)
green = CvScalar.new(0,255,0)

img_contours = src_img.draw_contours(contours, black, blue, 1, {thickness:2, line_type:8})

img_contours.save_image("contours.png")



# 画像の切り抜きがしたい #pythonの書き方(top:bottom, left:right)
# cutted_img = src_img[0, 100, 0, 100]
# rubyではこれでは無理な様子(Rmagickとかminimagickで切り抜きできる様子..)
# ほしいのは輪郭内のエリアの色だから、画像を切り抜く必要はないけど...
# pythonの方がやりやすそうだな...

