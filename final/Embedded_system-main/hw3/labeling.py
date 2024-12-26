import cv2
import numpy as np

def label_similar_areas(image, mask):
    """
    基於顏色資訊標記相似區域，並排除天空
    
    Parameters:
    image: 原始圖片
    mask: 二值化遮罩
    
    Returns:
    colored_image: 標記後的圖片
    """
    # 轉換圖片到HSV色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    fixed_color = (0, 0, 255)  # 紅色
    # 定義天空的顏色範圍 (在HSV空間中)
    # 可以根據實際情況調整這些值
    sky_lower = np.array([90, 50, 50])  # 偏淺藍色的下限
    sky_upper = np.array([130, 255, 255])  # 偏淺藍色的上限
    
    # 創建天空的遮罩
    sky_mask = cv2.inRange(hsv, sky_lower, sky_upper)
    
    # 將天空區域從原始遮罩中移除
    mask[sky_mask > 0] = 0
    
    # 進行連通區域標記
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    # 創建輸出圖像
    colored_image = image.copy()
    
    # 為每個標記區域著色
    for label in range(1, num_labels):  # 從1開始以跳過背景(0)
        # 獲取當前標記的遮罩
        current_mask = labels == label
        
        # 計算區域的平均HSV值
        region_hsv = hsv[current_mask]
        avg_hsv = np.mean(region_hsv, axis=0)
        
        # 如果區域不是天空色（可以根據需要調整條件）
        if not (sky_lower[0] <= avg_hsv[0] <= sky_upper[0] and 
                sky_lower[1] <= avg_hsv[1] <= sky_upper[1] and 
                sky_lower[2] <= avg_hsv[2] <= sky_upper[2]):
            
            # 根據區域大小決定是否標記
            if np.sum(current_mask) > 100:  # 可以調整這個閾值
                # 將固定顏色應用到圖像
                colored_image[current_mask] = fixed_color
    
    return colored_image
