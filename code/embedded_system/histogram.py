import numpy as np
import cv2
from matplotlib import pyplot as plt

def calculate_histogram(image_path, num_bins=256, save_hist=False):
    """
    計算圖像灰度直方圖並找出頻率最高的灰度值。
    
    :param image_path: 圖像路徑
    :param num_bins: 直方圖的分箱數
    :param save_hist: 是否保存直方圖圖片
    :return: 頻率最高的前三個灰度值及其直方圖
    """
    # 讀取灰度圖像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or invalid format.")

    # 計算直方圖
    hist = cv2.calcHist([img], [0], None, [num_bins], [0, 256])

    # 找出頻率最高的前三個灰度值
    top_3_values = np.argsort(hist.ravel())[::-1][:3]

    # 打印結果
    print("Top 3 Gray Values and Frequencies:")
    for i, value in enumerate(top_3_values):
        print(f"Rank {i+1}: Gray Value = {value}, Frequency = {int(hist[value])}")

    # 顯示直方圖
    plt.figure(figsize=(10, 5))
    plt.plot(hist, color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Gray Value')
    plt.ylabel('Frequency')
    if save_hist:
        plt.savefig('histogram_output.png')  # 保存圖片
    plt.show()

    return top_3_values, hist

# 使用範例
top_3_values, histogram = calculate_histogram('lbp2_result.jpg', save_hist=True)