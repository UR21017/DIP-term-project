#./linear_interpolation/img/2-1.jpg
import cv2
import numpy as np

class FocalStackRefocusing:
    def __init__(self, image_paths):
        self.images = [cv2.imread(path) for path in image_paths]
        self.image_index = 0

        # 創建窗口並設置鼠標回調函數
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_click)

    def calculate_sharpness(self, image, point):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.abs(laplacian).var()
        return sharpness

    def calculate_focus_map(self, image):
        h, w, _ = image.shape
        focus_map = np.zeros((h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                focus_map[i, j] = self.calculate_sharpness(image, (j, i))

        return focus_map

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            click_point = (x, y)
            print(f"Clicked point: ({x}, {y})")

            # 計算點擊位置的清晰度分佈
            focus_map = self.calculate_focus_map(self.images[self.image_index])

            # 找到最清晰的區域
            max_sharpness_index = np.unravel_index(np.argmax(focus_map), focus_map.shape)
            focal_point = (max_sharpness_index[1], max_sharpness_index[0])

            print(f"Focal point: {focal_point}")

            # 切換到最清晰的區域
            self.image_index = np.argmax(focus_map)
            cv2.imshow("Image", self.images[self.image_index])

    def run(self):
        while True:
            cv2.imshow("Image", self.images[self.image_index])

            # 等待按鍵事件
            key = cv2.waitKey(0) & 0xFF

            # 退出程序
            if key == 27:  # ESC key
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替換為你的圖片路徑
    image_paths = ["./linear_interpolation/img/2-1.jpg", "./linear_interpolation/img/2-2.jpg", "./linear_interpolation/img/2-3.jpg"]

    focal_stack_refocusing = FocalStackRefocusing(image_paths)
    focal_stack_refocusing.run()
