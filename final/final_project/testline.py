import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

def count_lines(image, offset=50):
    """
    根據長度過濾掉較短的線條，並統計直線數量。
    
    參數:
    - image: 圖像數據（已二值化處理）
    - offset: 設置過濾條件的閾值
    
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
    
    # 根據設置的閾值過濾掉過短的區域
    length_threshold = offset  # 設置過濾的閾值
    line_count = sum(1 for l in lengths if l >= length_threshold)  # 統計符合條件的直線數量
    
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
    
    # 定義裁剪區域的範圍（這裡假設保留中間區域的範圍，您可以根據需要調整）
    height, width = imagex.shape  # 讀取圖像的尺寸
    crop_margin = 3 # 設置裁剪的邊距，這裡是 50 像素，您可以根據需要調整
    
    # 裁剪圖像（只保留中間區域）
    croppedx = imagex[crop_margin:height-crop_margin, crop_margin:width-crop_margin]
    croppedy = imagey[crop_margin:height-crop_margin, crop_margin:width-crop_margin]
    
    # 顯示裁剪後的圖片（可視化）
    cv2.imshow('Cropped Sobel X', croppedx)
    cv2.imshow('Cropped Sobel Y', croppedy)
    
    # 二值化處理
    _, binaryx = cv2.threshold(croppedx, 127, 255, cv2.THRESH_BINARY)
    _, binaryy = cv2.threshold(croppedy, 127, 255, cv2.THRESH_BINARY)
    
    # 創建一個 11x11 的結構元素（結構元素的大小可根據需求調整）
    kernel = np.ones((3, 3), np.uint8)
    
    # 進行二值膨脹操作，將圖像中的前景區域擴大
    dilatedx = binary_dilation(binaryx, structure=kernel).astype(np.uint8)
    dilatedy = binary_dilation(binaryy, structure=kernel).astype(np.uint8)
    
    # 正常化圖像範圍以便保存
    dilatedx_normalized = cv2.normalize(dilatedx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dilatedy_normalized = cv2.normalize(dilatedy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 保存膨脹後的圖像
    cv2.imwrite('sobelx_dilated.jpg', dilatedx_normalized)
    cv2.imwrite('sobely_dilated.jpg', dilatedy_normalized)

    
    # 計算膨脹後符合條件的直線數量
    line_countx = count_lines(dilatedx, offset=22)  # 這裡可以調整offset來過濾短的線條
    line_county = count_lines(dilatedy, offset=22)
    
    # 計算總的直線數量
    total_line_count = line_countx + line_county
    print(f'Total lines count (after dilation and filtering): {total_line_count}')
    
    # 等待按鍵退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()
