import numpy as np
from PIL import Image

def get_pixel_value(img, center, x, y):
    """
    獲取像素值，如果座標超出圖像範圍則返回中心點的值
    """
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def calculate_lbp(img, x, y):
    """
    計算單個像素的LBP值
    """
    center = img[x][y]
    val_ar = []
    
    # 順時針收集8個鄰居的值
    val_ar.append(get_pixel_value(img, center, x-1, y-1))
    val_ar.append(get_pixel_value(img, center, x-1, y))
    val_ar.append(get_pixel_value(img, center, x-1, y+1))
    val_ar.append(get_pixel_value(img, center, x, y+1))
    val_ar.append(get_pixel_value(img, center, x+1, y+1))
    val_ar.append(get_pixel_value(img, center, x+1, y))
    val_ar.append(get_pixel_value(img, center, x+1, y-1))
    val_ar.append(get_pixel_value(img, center, x, y-1))
    
    # 將二進制值轉換為十進制
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def get_lbp_image(image_path):
    """
    計算整張圖片的LBP特徵
    """
    # 讀取圖片並轉為灰度圖
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # 創建輸出圖像陣列
    height, width = img_array.shape
    lbp_array = np.zeros((height-2, width-2), dtype=np.uint8)
    
    # 計算每個像素的LBP值
    for i in range(1, height-1):
        for j in range(1, width-1):
            lbp_array[i-1][j-1] = calculate_lbp(img_array, i, j)
    
    return lbp_array

def get_lbp_histogram(lbp_array):
    """
    計算LBP特徵的直方圖
    """
    histogram = np.zeros(256, dtype=np.int32)
    for i in range(lbp_array.shape[0]):
        for j in range(lbp_array.shape[1]):
            histogram[lbp_array[i][j]] += 1
    return histogram

# 使用範例
def main_lbp():
    # 替換成您的圖片路徑
    image_path = "sobel_result.jpg"
    
    # 獲取LBP特徵圖
    lbp_image = get_lbp_image(image_path)
    
    # 計算LBP直方圖
    histogram = get_lbp_histogram(lbp_image)
    
    # 將LBP特徵圖保存
    lbp_img = Image.fromarray(lbp_image)
    lbp_img.save("lbp_result.jpg")
    
    return lbp_image, histogram

main_lbp()