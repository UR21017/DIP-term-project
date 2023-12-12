import tkinter as tk
from PIL import Image, ImageTk

class ImageSwitcherApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Switcher")

        self.images = [
            Image.open("./linear_interpolation/img2/1.jpg"),  # 替换为你的图片路径
            Image.open("./linear_interpolation/img2/2.jpg"),
            Image.open("./linear_interpolation/img2/3.jpg"),
            Image.open("./linear_interpolation/img2/4.jpg"),
            Image.open("./linear_interpolation/img2/5.jpg")
        ]

        self.current_image_index = 0

        # 调整图像大小并裁剪，适应屏幕
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        for i, img in enumerate(self.images):
            self.images[i] = self.resize_and_crop_image(img, screen_width, screen_height)

        self.displayed_image = ImageTk.PhotoImage(self.images[self.current_image_index])

        self.canvas = tk.Canvas(self.master, width=screen_width, height=screen_height)
        self.canvas.pack()

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.displayed_image)

        # 设置鼠标点击事件
        self.canvas.bind("<Button-1>", self.switch_image)

    def resize_and_crop_image(self, img, target_width, target_height):
        # 调整图像大小
        img.thumbnail((target_width, target_height))
        
        # 计算裁剪框的位置
        left_margin = (img.width - target_width) / 2
        top_margin = (img.height - target_height) / 2
        right_margin = (img.width + target_width) / 2
        bottom_margin = (img.height + target_height) / 2

        # 裁剪图像
        img_cropped = img.crop((left_margin, top_margin, right_margin, bottom_margin))
        
        return img_cropped

    def switch_image(self, event):
        # 切换到下一张图
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.displayed_image = ImageTk.PhotoImage(self.images[self.current_image_index])

        # 更新Canvas上的图像
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.displayed_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSwitcherApp(root)
    root.mainloop()
