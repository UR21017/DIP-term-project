# Seamless Refocusing

原本在切換由近至遠對焦的圖片時，看起來整體縮小，切換由遠至近的對焦的圖片，看起來則是整體放大。透過以下程式，可以達到切換焦點時不會產生縮放的效果。
使用SIFT算法進行圖像對齊和焦點合併。該腳本可以加載一系列圖片，通過點擊其中一張圖片選擇感興趣區域，然後找到其他圖片中對應區域的最佳焦點，最終生成一張合併後的圖像。
- [成果](https://youtu.be/qVbRndE4l4c)

## 使用方法
1. 將待處理的圖片放入指定文件夾（默認為"./pictures/"）中，圖片格式應為JPEG格式。
2. 運行腳本，點擊要選擇感興趣區域的圖片上的目標區域（左鍵點擊）。
3. 程序會自動在其他圖片中找到相似區域的最佳焦點，並將合併後的圖像顯示出來。

## 畫面

> 可提供 1~3 張圖片，讓觀看者透過 README 了解整體畫面

![範例圖片 1](https://github.com/UR21017/DIP-term-project/blob/main/pictures/picture1.jpg)
![範例圖片 2][(https://fakeimg.pl/500/)](https://youtu.be/qVbRndE4l4c)

## 安裝

> 請務必依據你的專案來調整內容。

確保安裝了以下Python庫：

- matplotlib
- numpy
- opencv-python

### 函數說明
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
