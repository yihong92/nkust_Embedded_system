import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_pixel_value(img, center, x, y):
    """
    獲取像素值，如果座標超出圖像範圍則返回中心點的值
    """
    if 0 <= x < img.shape[0] and 0 <= y < img.shape[1] and img[x][y] >= center:
        return 1
    return 0

def calculate_lbp(img, x, y):
    """
    計算單個像素的LBP值
    """
    center = img[x][y]
    val_ar = [
        get_pixel_value(img, center, x-1, y-1),
        get_pixel_value(img, center, x-1, y),
        get_pixel_value(img, center, x-1, y+1),
        get_pixel_value(img, center, x, y+1),
        get_pixel_value(img, center, x+1, y+1),
        get_pixel_value(img, center, x+1, y),
        get_pixel_value(img, center, x+1, y-1),
        get_pixel_value(img, center, x, y-1)
    ]
    
    # 二進制轉十進制
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    return sum(val_ar[i] * power_val[i] for i in range(len(val_ar)))

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

# 使用範例
def main():
    image_path = "picture/sobel_result.jpg"  # 替換為您要處理的圖片路徑
    
    # 獲取LBP特徵圖
    lbp_image = get_lbp_image(image_path)
    
    # 保存LBP特徵圖
    lbp_img = Image.fromarray(lbp_image)
    lbp_img.save("picture/lbp_result.jpg")  # 保存為新的圖片

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(lbp_image, cmap='gray')
    plt.title('LBP Image')
    plt.axis('off')


    plt.tight_layout()
    plt.show()
    
    return lbp_image

# 執行程式
if __name__ == "__main__":
    lbp_image = main()