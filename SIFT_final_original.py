import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backend_bases import MouseButton
import glob

folder_path = "./pictures/"
image_paths = glob.glob(f"{folder_path}*.jpg")


MIN_MATCH_COUNT = 10

def SIFT(foreground, background):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(foreground, None)
    kp2, des2 = sift.detectAndCompute(background, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w, c = foreground.shape
        pts = np.float32([[0,0], [0, h-1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        #background = cv2.polylines(background, [np.int32(dst)], True, (255, 255, 255), 3, cv2.LINE_AA) #frame
        mask = np.zeros_like(background)
        cv2.fillPoly(mask, [np.int32(dst)], 255)
        pts_rect = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        M_inv = cv2.getPerspectiveTransform(dst, pts_rect)
        background = cv2.warpPerspective(background, M_inv, (w, h)) # adjust size inside the frame
        cv2.polylines(background, [np.int32(pts_rect)], isClosed=True, color=(255, 255, 255), thickness=2)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)
        
        #img3 = cv2.drawMatches(foreground,kp1,background,kp2,good,None,**draw_params)
        
        #plt.imshow(img3, 'gray'),plt.show()
        
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    
    return background



def load_image(standand, targetImage):
    foreground = cv2.imread(standand)
    background = cv2.imread(targetImage)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    processed_image = SIFT(foreground,  background)
    

    return processed_image

def calculate_image_quality(image):
    return np.var(cv2.Laplacian(image, cv2.CV_64F))

def find_best_focus_image(directory, click_x, click_y, window_size=50):
    best_quality = -1
    for image in directory:
  
        x= max(0, int(click_x) - window_size)
        y = max(0,int(click_y) - window_size)

        window = image[y:min(y + window_size * 2, image.shape[0]), x:min(x + window_size * 2, image.shape[1])]
        quality = calculate_image_quality(window)

        if quality > best_quality:
            best_quality = quality
            best_image = image

    return best_image


def on_click(event):
    if event.button is MouseButton.LEFT:
        global click_x, click_y

        click_x, click_y = event.xdata, event.ydata

        best_image = find_best_focus_image(directory, click_x, click_y)


        ax.clear()
        ax.imshow(best_image)  
        plt.axis("off"),plt.draw()



directory = []

for image_path in image_paths:

    directory.append(load_image(image_paths[0], image_path))
    # cv2.imshow("result",load_image(image_paths[0], image_path))
    # cv2.waitKey(0)

# print(len(directory))

fig, ax = plt.subplots()
ax.imshow(directory[0]) 

plt.connect('button_press_event', on_click)

manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
plt.axis("off"),plt.show()