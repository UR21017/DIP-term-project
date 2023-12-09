import cv2

# 讀取圖片
image = cv2.imread('images/2-3.jpg')  # 替換為實際的圖片路徑

# 指定放大倍數
scale_factor = 1.05  # 替換為實際的放大倍數

# 計算放大後的寬度和高度
enlarged_width = int(image.shape[1] * scale_factor)
enlarged_height = int(image.shape[0] * scale_factor)

# 計算裁剪框的位置和大小
crop_x = max(0, (enlarged_width - image.shape[1]) // 2)
crop_y = max(0, (enlarged_height - image.shape[0]) // 2)
crop_width = min(enlarged_width, image.shape[1])
crop_height = min(enlarged_height, image.shape[0])

# 進行裁剪和放大操作
cropped_enlarged_image = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
result_image = cv2.resize(cropped_enlarged_image, (image.shape[1], image.shape[0]))

# 顯示原始圖片、裁剪並放大後的圖片
cv2.imshow('Original Image', image)
cv2.imshow('Cropped and Enlarged Image', result_image)
cv2.imwrite("./results/2.jpg", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
