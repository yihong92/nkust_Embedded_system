import cv2
import numpy as np

# 讀取輸入圖像
img = cv2.imread('./photo/test_2.jpg')  # 讀取名稱為'test.jpg'的圖像文件

# 將圖像轉換為灰度圖像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 將彩色圖像轉換為灰度圖像

# 進行二值化處理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 將灰度圖像進行二值化處理

# 執行形態學操作
kernel = np.ones((5, 5), np.uint8)  # 創建一個 5x5 的核
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 進行開運算去除噪聲

# 進行二值化處理
ret, thresh = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)  # 將灰度圖像進行二值化處理


# 找到輪廓
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 發現圖像中的輪廓

# 在空白圖像上繪製輪廓
drawing = np.zeros_like(img)  # 創建一個與原始圖像大小相同的空白圖像
cv2.drawContours(drawing, contours, -1, (0, 255, 0), 2)  # 用綠色線條繪製所有的輪廓

# 保存輸出圖像
cv2.imwrite('./photo/output_image.jpg', drawing)  # 將繪製了輪廓的圖像保存為'output_image.jpg'
#cv2.subtract()