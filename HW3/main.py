import cv2
import sobel
import lbp_notcv2
import search
import histogram
import one_norm_dist
import label
import hsv
import numpy as np
from PIL import Image

def main():
    patch_size = 16
    # 讀取影像
    image      = cv2.imread('picture/road.jpg')
    #hsv
    lower = np.array([0, 0, 0])   # HSV的下限
    upper = np.array([180, 200, 100])  # HSV的上限 
    image_hsv  = hsv.hsv(image,lower,upper)

    image_gray = cv2.imread('picture/hsv.jpg',0)

    

    #sobel
    img_sobel = sobel.apply_sobel(image_hsv)
    #lbp
    img_lbp   = lbp_notcv2.main()

    #histogram
    mask = np.zeros_like(image_gray)

    his      = histogram.calculate_histogram(img_lbp)
    top3     = histogram.find_top_three(his)
    th        = int(sum(top3)/3)
    print(th)

    rows, cols = img_lbp.shape
    for i in range(0, rows, patch_size):
        for j in range(0, cols, patch_size):
            patch1 = img_lbp[i:i+patch_size, j:j+patch_size]
            if i+patch_size < rows and j+patch_size < cols:
                # 與右邊的區塊進行比較
                patch2 = img_lbp[i:i+patch_size, j+patch_size:j+2*patch_size]
                hist1 = histogram.calculate_histogram(patch1)
                hist2 = histogram.calculate_histogram(patch2)               
                if one_norm_dist.calculate_1_norm_distance(hist1, hist2,th) == 1:
                    mask[i:i+patch_size, j:j+patch_size] = 1  # 標記需要上色的區域


    final_image = label.label_similar_areas(image, mask, color=(0, 0, 255), alpha=0.6)
    final_picture = Image.fromarray(final_image)
    final_picture.save("picture/final.jpg")  # 保存為新的圖片
    cv2.imshow('Colored Image', final_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()