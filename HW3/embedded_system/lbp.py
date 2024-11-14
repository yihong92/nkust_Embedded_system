import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern

def apply_lbp(input_image_path, points=8, radius=1):
    # 讀取輸入圖片
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # 計算LBP
    lbp = local_binary_pattern(img, points, radius, method="uniform")
    
    # 將LBP圖像轉換為uint8類型
    lbp_result = np.uint8((lbp / lbp.max()) * 255)
    
    # 顯示結果
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Input Image (Sobel Result)')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(lbp_result, cmap='gray')
    plt.title('LBP Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存結果
    cv2.imwrite('lbp_result.jpg', lbp_result)
    
    return lbp_result

# 使用函數
# 假設Sobel處理後的圖片被保存為'sobel_result.jpg'
result = apply_lbp('sobel_result.jpg')
