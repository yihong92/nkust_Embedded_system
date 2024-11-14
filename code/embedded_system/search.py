import cv2
import numpy as np
from matplotlib import pyplot as plt

def split_image_into_patches(image, patch_size):
    """
    將影像分割成多個小區塊並存儲在列表中。
    
    參數:
    image (numpy.ndarray): 輸入的灰度影像
    patch_size (int): 每個小區塊的大小
    
    返回:
    list: 包含所有小區塊的列表
    """
    patches = []
    rows, cols = image.shape
    
    for i in range(0, rows, patch_size):
        for j in range(0, cols, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    
    return patches

def main():
    # 讀取影像
    image_file = 'lbp_result.jpg'
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: unable to load image '{image_file}'")
    
    # 設置分割區塊大小
    patch_size = 32  # 例如，16x16 的區域

    # 將影像分割成小區塊
    patches = split_image_into_patches(image, patch_size)

    # 顯示原始影像和部分小區塊
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    for idx, patch in enumerate(patches[:10]):  # 只顯示前10個小區塊
        plt.subplot(3, 10, idx + 11)
        plt.imshow(patch, cmap='gray')
        plt.title(f'Patch {idx + 1}')
        plt.axis('off')

    plt.suptitle('Original Image and Patches')
    plt.show()

if __name__ == '__main__':
    main()