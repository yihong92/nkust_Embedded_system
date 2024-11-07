import numpy as np
import cv2
from matplotlib import pyplot as plt

def calculate_histogram(image_path, num_bins=256):
    # 讀取圖像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 計算直方圖
    hist = cv2.calcHist([img], [0], None, [num_bins], [0, 256])
    
    # 找出直方圖中頻率最高的前三大值的索引（灰度值）
    top_3_values = np.argsort(hist.ravel())[::-1][:3]
    
    # 輸出前三大值的灰度值和頻率
    print("Top 3 Gray Values and Frequencies:")
    for i, value in enumerate(top_3_values):
        print(f"Rank {i+1}: Gray Value = {value}, Frequency = {int(hist[value])}")
    
    # 顯示直方圖
    plt.figure(figsize=(10, 5))
    plt.plot(hist, color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Gray Value')
    plt.ylabel('Frequency')
    plt.show()
    
    return top_3_values, hist

# 使用函數
# 假設 LBP 結果圖片被保存為 'lbp_result.jpg'
top_3_values, histogram = calculate_histogram('lbp_result.jpg')