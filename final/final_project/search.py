import cv2
import numpy as np
import pandas
from pandas import DataFrame

def search(input_image_path, area_min=3100, area_max=7000):
    """
    從圖像中檢測輪廓並篩選符合條件的區域。

    參數:
    - input_image_path: 輸入圖像的路徑。
    - area_min: 篩選輪廓的最小面積（默認值為3100）。
    - area_max: 篩選輪廓的最大面積（默認值為7000）。

    返回:
    - grid_contours: 符合條件的輪廓列表。
    """
    # 讀取圖像並轉換為灰度圖像
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("無法讀取圖像，請檢查路徑是否正確。")
        return []

    # 自適應二值化處理
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # 找到輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 篩選符合條件的輪廓
    grid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min < area < area_max:  # 篩選面積在範圍內的輪廓
            grid_contours.append(cnt)

    # 在圖像上繪製篩選後的輪廓
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, grid_contours, -1, (0, 255, 0), 2)  # 綠色輪廓
    '''
    # 顯示結果圖像
    cv2.imshow("Detected Grid Contours", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # 返回篩選後的輪廓
    print(f"找到 {len(grid_contours)} 個符合條件的輪廓。")
    return grid_contours


def sort_contours(grid_contours):
    # 將輪廓按y座標分組（考慮一定的容差）
    def get_row_index(y, tolerance=20):
        return int(y / tolerance)
        
    # 獲取所有矩形的邊界
    boxes = [cv2.boundingRect(c) for c in grid_contours]
    
    # 按y座標分組
    rows = {}
    for i, (x, y, w, h) in enumerate(boxes):
        row_idx = get_row_index(y)
        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append((grid_contours[i], x))
    
    # 對每一行按x座標排序
    sorted_contours = []
    for row_idx in sorted(rows.keys()):
        row_contours = rows[row_idx]
        row_contours.sort(key=lambda x: x[1])  # 按x座標排序
        sorted_contours.extend([cnt for cnt, _ in row_contours])
    
    return sorted_contours



if __name__ == "__main__":
    labeled_image = search()
