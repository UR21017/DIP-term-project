import cv2
import numpy as np
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config

# 設定Mask R-CNN的配置
class InferenceConfig(Config):
    NAME = "foreground_background"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # 前景和背景

config = InferenceConfig()

# 建立Mask R-CNN模型
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')

# 載入預訓練的權重（COCO資料集）
model.load_weights('mask_rcnn_coco.h5', by_name=True)

# 初始化相機
cap = cv2.VideoCapture(0)

while True:
    # 讀取當前幀
    ret, frame = cap.read()

    # 預測Mask R-CNN結果
    results = model.detect([frame], verbose=0)
    r = results[0]

    # 提取前景區域
    foreground_mask = r['masks'][:, :, 0]
    foreground = frame.copy()
    foreground[foreground_mask == 0] = 0

    # 提取背景區域
    background = frame.copy()
    background[foreground_mask > 0] = 0

    # 顯示結果
    cv2.imshow('Foreground', foreground)
    cv2.imshow('Background', background)

    # 按 'q' 鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()