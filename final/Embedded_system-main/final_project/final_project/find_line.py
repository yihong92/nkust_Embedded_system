import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
import split_image

def count_lines(image, offset=20):
    """
    找出最小的兩個長度，並用來設置動態閾值，統計直線數量。
    
    參數:
    - image: 圖像數據（已二值化處理）
    - offset: 在最小兩個長度基礎上加的偏移量
    
    回傳:
    - 直線數量
    """
    # 確認 image 不為 None
    if image is None:
        print("Error: 圖像數據為 None。")
        return
    
    # 連通區域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    # 計算所有區域的長度（寬度和高度）
    lengths = []
    for i in range(1, num_labels):  # 0 是背景，不需要處理
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        lengths.append(max(width, height))  # 取寬和高中的較大值作為長度
    
    '''
    # 找出最小的兩個長度
    if len(lengths) >= 2:
        sorted_lengths = sorted(lengths)  # 排序長度列表
        min1, min2 = sorted_lengths[:2]  # 取出最小的兩個長度
        length_threshold = (min1 + min2)/2 + offset  # 動態閾值 = 最小兩個長度相加 + 偏移量
    else:
        print("未檢測到足夠的連通區域。")
        return 0
        
    print(f"最小的兩個長度: {min1}, {min2}")
    print(f"動態計算的長度閾值: {length_threshold}")
    '''
    length_threshold = offset
    # 統計符合條件的直線數量
    line_count = sum(1 for l in lengths if l >= length_threshold)
    
    print(f"統計直線數量: {line_count}")
    return line_count

# 主程序
if __name__ == "__main__":
    # 設定圖像文件路徑
    image_pathx = './photo/sobelx.jpg'  # 確保此文件存在於指定路徑
    image_pathy = './photo/sobely.jpg'
    
    # 讀取並進行灰度處理
    imagex = cv2.imread(image_pathx, cv2.IMREAD_GRAYSCALE)
    imagey = cv2.imread(image_pathy, cv2.IMREAD_GRAYSCALE)
    
    # 二值化處理
    _, binaryx = cv2.threshold(imagex, 127, 255, cv2.THRESH_BINARY)
    _, binaryy = cv2.threshold(imagey, 127, 255, cv2.THRESH_BINARY)    
    # 創建一個 5x5 的結構元素
    kernel = np.ones((11, 11), np.uint8)
    '''
    # 進行二值膨脹操作
    openedx = binary_dilation(binaryx, structure=kernel).astype(np.uint8)
    openedy = binary_dilation(binaryy, structure=kernel).astype(np.uint8)   
    cv2.imwrite('sobelx_mor.jpg',openedx)
    cv2.imwrite('sobely_mor.jpg',openedy)
    # 計算符合條件的直線數量
    # '''
    line_countx = count_lines(binaryx, offset=20)
    line_county = count_lines(binaryy, offset=20)
    line_count = line_countx + line_county
    print(f'total = {line_count}')
