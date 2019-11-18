require 'opencv'
include OpenCV


#画像を読み込む
image_folder = '../image/'
image_file  = 'shikaku.png'
src_img = CvMat.load(image_folder + image_file)  #matrixオブジェクトにロード

 
#大きく２値化
gray_img = src_img.BGR2GRAY #グレースケールで取得
#しきい値基準の２値画像の作成. http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
bin_img = gray_img.threshold(200, 255, :binary)  #白と判断する値の範囲が200〜255ということ。らしい
canny_img = src_img.BGR2GRAY.canny(120,200) #輪郭の取得..
 
# GUI::Window.new('src').show(src_img)
# GUI::Window.new('gray').show(gray_img)
GUI::Window.new('bin').show(bin_img)
# GUI::Window.new('canny').show(canny_img) 
 
GUI::wait_key
