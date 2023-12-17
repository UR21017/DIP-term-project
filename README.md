# Seamless Refocusing

此程式使用SIFT算法進行圖像對齊和焦點合併，可以輸入一組焦點不同的圖片，透過點擊圖片中感興趣區域，找到其他圖片中對應區域的最佳焦點。達到切換焦點時，視覺上不會感受到圖片縮放的效果。
- [成果](https://youtu.be/IJC7GbT9xuE)

## 題目說明
Allowing the change of focal point while browsing a photo is an attractive feature for smartphone users because it avoids reshooting the photo. Focal-stack refocusing is an appropriate approach to implement such a feature for your smartphone. It takes a sequence of photos (the focal stack) of a scene by sweeping the lens and selects from the focal stack the sharpest photo corresponding to the focal point specified by the user. However, switching from one focal point to another may cause an annoying radial expansion or contraction to the displayed photos. The goal of this term project is to resolve the expansion/contraction issue by designing an effective and efficient algorithm.
> 由近至遠對焦:圖片逐漸縮小
![由近至遠對焦:圖片逐漸縮小](https://github.com/UR21017/DIP-term-project/blob/main/images%20(2).gif)

## 使用方法
1. 將待處理的圖片放入指定文件夾（默認為"./pictures/"）中，圖片格式應為JPEG格式。
2. 運行腳本，點擊要選擇感興趣區域的圖片上的目標區域（左鍵點擊）。
3. 程序會自動在其他圖片中找到相似區域的最佳焦點，並將合併後的圖像顯示出來。

## 安裝
- matplotlib
- numpy
- opencv-python

## 函數說明
`SIFT(foreground, background)`
使用SIFT算法進行圖像匹配和對齊。在足夠匹配點的情況下，計算圖像變換矩陣，然後通過透視變換將背景圖像中的區域與前景圖像對齊。

`load_image(standand, targetImage)`
加載標準圖像和目標圖像，將它們轉換為RGB格式，並調用SIFT函數進行對齊操作。

`calculate_image_quality(image)`
計算圖像質量，使用拉普拉斯算子的方差作為圖像清晰度的度量。

`find_best_focus_image(directory, click_x, click_y, window_size=50)`
在圖像目錄中查找與點擊位置相對應的最佳焦點圖像。通過計算窗口內圖像的清晰度來評估焦點。

`on_click(event)`
點擊事件處理函數，當左鍵點擊時，獲取點擊坐標並調用find_best_focus_image函數，然後在Matplotlib圖形中顯示最佳焦點圖像。

## 參考資料
- [SIFT原理](https://zh.wikipedia.org/zh-tw/%E5%B0%BA%E5%BA%A6%E4%B8%8D%E8%AE%8A%E7%89%B9%E5%BE%B5%E8%BD%89%E6%8F%9B)
- [Feature Matching + Homography to find objects](https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html)
