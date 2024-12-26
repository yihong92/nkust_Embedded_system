import cv2
import numpy as np
import sobel
import hsv
import search
import lbp
import histogram 
import one_norm_dist
import labeling
def main():
    radius = 1
    n_points = 8  
    patch_size = 8
    # 讀取影像
    image = cv2.imread('test.jpg')
    
    #HSV(將深色部分轉變為黑色)(將馬路的顏色統一為黑色)
    lower = np.array([0, 0, 0])   # HSV的下限
    upper = np.array([180, 200, 100])  # HSV的上限 
    hsv_img = hsv.hsv(image, lower, upper)
    cv2.imwrite('hsv.jpg',hsv_img)
    image_gray = cv2.imread('hsv.jpg',0)
    height, width = image_gray.shape
    
    #sobel
    Sobel_img = sobel.sobel_and_morphology(image_gray)
    cv2.imwrite('sobel.jpg',Sobel_img)
    
    #search
    noise_threshold = 1000 # 定義噪點區域大小的閾值
    search_img = search.bfs_remove_noise_optimized(Sobel_img, noise_threshold)
    cv2.imwrite('search.jpg',search_img)
    
    #lbp
    lbp_img = lbp.calculate_lbp(search_img, radius, n_points)
    cv2.imwrite('lbp.jpg',lbp_img)
    
    #histogram
    
    his = histogram.calculate_histogram(image)
    hist = histogram.plot_histogram(his)
    #histogram找前三大(設定閥值)
    top3 = histogram.find_top_three(his)
    th = int(sum(top3)/3)

    #1_norm_dist
    mask = np.zeros_like(image_gray)  # 創建與影像相同尺寸的 mask
    rows, cols = lbp_img.shape
    for i in range(0, rows, patch_size):
        for j in range(0, cols, patch_size):
            patch1 = lbp_img[i:i+patch_size, j:j+patch_size]
            if i+patch_size < rows and j+patch_size < cols:
                # 與右邊的區塊進行比較
                patch2 = lbp_img[i:i+patch_size, j+patch_size:j+2*patch_size]
                hist1 = histogram.calculate_histogram(patch1)
                hist2 = histogram.calculate_histogram(patch2)               
                if one_norm_dist.calculate_1_norm_distance(hist1, hist2,th) == 1:
                    mask[i:i+patch_size, j:j+patch_size] = 1  # 標記需要上色的區域

    #labeling
    colored_img = labeling.label_similar_areas(image, mask)

    cv2.imwrite('colored_patch_size=8.jpg',colored_img)    
    
    # 顯示
    # 原始影像和著色後的影像
    cv2.imshow('Colored Image', colored_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
