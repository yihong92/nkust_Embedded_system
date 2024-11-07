import numpy as np
import cv2
from matplotlib import pyplot as plt

def apply_sobel(image_path):
    # 讀取圖片
    img = cv2.imread(image_path)
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 計算 X 方向的 Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # 計算 Y 方向的 Sobel
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 計算梯度幅值
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # 將梯度幅值轉換為0-255的範圍
    magnitude = np.uint8(np.absolute(magnitude))
    
    # 創建圖像顯示
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(np.absolute(sobelx), cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(np.absolute(sobely), cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存結果
    cv2.imwrite('sobel_result.jpg', magnitude)
    
    return magnitude

# 使用函數
result = apply_sobel('road.jpg')