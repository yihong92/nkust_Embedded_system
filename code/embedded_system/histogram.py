import cv2
import numpy as np
from matplotlib import pyplot as plt

def calculate_histogram(image):
    # 計算直方圖
    image = image.astype(np.uint8)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # 歸一化直方圖
    #hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist

def plot_histogram(hist, output_path='.picture/histogram.jpg'):
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
    top_three_values = hist[top_three_indices] # 找到前三大值的值並展平 
    # 將結果轉換為可讀格式 
    #top_three = [(int(idx), int(val[0])) for idx, val in zip(top_three_indices, top_three_values)] 
    top_three = [int(idx) for  idx in top_three_indices] 
    #top_three = [int(val[0]) for val in top_three_values]
    print(top_three)

    return top_three
