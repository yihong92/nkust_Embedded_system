import cv2
import time
import numpy as np
import gray_img
import threshold
import morphology
import find_line
import sobel
import search
import find_line
import calculate_total_price
from scipy.ndimage import binary_dilation

# 記錄開始時間
start_time = time.time()

# 讀取輸入圖像
img = cv2.imread('./photo/test_all3.jpg')  # 讀取測試的圖片

# gray_img
gray = gray_img.bgr_to_gray(img)

# threshold(手刻)
binary = threshold.threshold(gray, 127, 255)

# morphology形態學去除文字(手刻)
kernel = np.ones((5, 5), np.uint8)  # 創建一個 5x5 的核
morphology_image = morphology.morphological_opening(binary // 255, kernel)

# 進行二值化處理
binary = threshold.threshold(morphology_image, 127, 255)
# 以下程式碼為用來調整search參數用的
'''
# 找到輪廓
contours, hierarchy = cv2.findContours(opened_custom, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 發現圖像中的輪廓

# 在空白圖像上繪製輪廓
drawing = np.zeros_like(img)  # 創建一個與原始圖像大小相同的空白圖像
cv2.drawContours(drawing, contours, -1, (0, 255, 0), 2)  # 用綠色線條繪製所有的輪廓

# 保存輸出圖像
cv2.imwrite('./photo/output_image.jpg', drawing)  # 將繪製了輪廓的圖像保存為'output_image.jpg'
'''
#search 找出格子座標
morphology_input = './photo/morphology.jpg'
xy_list = search.search(morphology_input, 3100, 7000)
xy_list = search.sort_contours(xy_list)
#每格價錢參數
price_list = [0,280,280,280,280,280,280,280,280,280,
              350,350,350,160,160,160,160,160,160,
              160,160,160,180,180,180,200,200,200,200,
              200,200,200,200,200,220,220,220,320,320,
              320,320,320,320,450,450,450,80,80,80,
              0,0,0,0,30,50,20,0,0
              ]

#print(xy_list)

# 創建一個空列表來存儲裁剪後的圖像
cropped_images = []

# 遍歷座標列表並裁剪圖像
for i, cnt in enumerate(xy_list):

    # 確保 cnt 是 NumPy 數組
    #cnt = np.array(cnt, dtype=np.int32)

    x,y,w,h = cv2.boundingRect(cnt)
    # 裁剪區域
    cropped_image = binary[y:y+h, x:x+w]
    cropped_images.append(cropped_image)
    
    # 可選：保存每個裁剪的圖像
    cv2.imwrite(f'./cropped/cropped_image_{i}.jpg', cropped_image)

#統計數量
all_line = []
for i in range(0,58):
    # 取得 Sobel 邊緣檢測結果
    sobelx, sobely = sobel.sobel(cropped_images[i])
    
    # 獲取圖像尺寸
    height, width = sobelx.shape
    
    # 設置裁剪邊距
    crop_margin = 3
    
    # 裁剪圖像
    croppedx = sobelx[crop_margin:height-crop_margin, crop_margin:width-crop_margin]
    croppedy = sobely[crop_margin:height-crop_margin, crop_margin:width-crop_margin]
    
    # 二值化處理
    _, binaryx = cv2.threshold(croppedx, 127, 255, cv2.THRESH_BINARY)
    _, binaryy = cv2.threshold(croppedy, 127, 255, cv2.THRESH_BINARY)
    
    # 創建結構元素用於膨脹操作
    kernel = np.ones((3, 3), np.uint8)
    
    # 執行二值膨脹
    dilatedx = binary_dilation(binaryx, structure=kernel).astype(np.uint8)
    dilatedy = binary_dilation(binaryy, structure=kernel).astype(np.uint8)
    
    # 計算線條數量
    x_line = find_line.count_lines(dilatedx, 23)
    y_line = find_line.count_lines(dilatedy, 23)
    total = x_line + y_line
    #print(i)
    #print(total)
    all_line.append(total)

price = calculate_total_price.calculate_total_price(all_line, price_list)

print('總金額',price)

# 記錄結束時間
end_time = time.time()

# 計算並顯示執行時間
execution_time = end_time - start_time
print(f"程式執行時間: {execution_time:.2f} 秒")