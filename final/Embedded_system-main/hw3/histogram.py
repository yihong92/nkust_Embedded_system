import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculate_histogram(image):

    # 確保影像是 uint8 格式
    image = image.astype(np.uint8)
    
    # 創建一個長度為 256 的陣列來儲存每個灰階值的出現次數
    hist = np.zeros(256, dtype=int)
    
    # 計算每個灰階值的出現次數
    for value in image.ravel():
        hist[value] += 1

    return hist


def plot_histogram(hist, output_path='./histogram.jpg'):
    # 繪製直方圖
    plt.figure(figsize=(10, 6))
    plt.plot(hist, color='black')
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def find_top_three(hist):
    top_three_indices = np.argsort(hist, axis=0)[-3:][::-1].flatten() # 找到前三大值的索引並展平 
    # 將結果轉換為可讀格式 
    #top_three = [(int(idx), int(val[0])) for idx, val in zip(top_three_indices, top_three_values)] 
    top_three = [int(idx) for  idx in top_three_indices] 
    #top_three = [int(val[0]) for val in top_three_values]

    return top_three

