import cv2
import numpy as np
import os

def calculate_image_quality(image):
    return np.var(cv2.Laplacian(image, cv2.CV_64F))

def find_best_focus_image(image_paths, click_x, click_y, window_size=50):
    best_quality = -1
    best_image_path = ""

    for image_path in image_paths:
        image = cv2.imread(image_path)

        x = max(0, click_x - window_size)
        y = max(0, click_y - window_size)
        window = image[y:min(y + window_size * 2, image.shape[0]), x:min(x + window_size * 2, image.shape[1])]

        quality = calculate_image_quality(window)

        if quality > best_quality:
            best_quality = quality
            best_image_path = image_path

    return best_image_path

def display_image(image_path):
    image = cv2.imread(image_path)
    cv2.imshow("Best Focus Image", image)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global click_x, click_y
        click_x, click_y = x, y
        best_image_path = find_best_focus_image(image_paths, click_x, click_y)
        display_image(best_image_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    folder_path = os.path.join(script_dir, "./images/set1")
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]

    sample_image = cv2.imread(image_paths[0])
    image_size = (sample_image.shape[1], sample_image.shape[0])

    click_x, click_y = -1, -1

    cv2.namedWindow("Best Focus Image", cv2.WINDOW_NORMAL)  # 使用可調整大小的窗口
    cv2.resizeWindow("Best Focus Image", *image_size)  # 設置窗口大小

    cv2.setMouseCallback("Best Focus Image", mouse_callback)

    while True:
        cv2.imshow("Best Focus Image", np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8))  # Placeholder image with the desired size
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
