import cv2
import numpy as np
import search
def mark_coordinates(image_path, coordinates):
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print("未能讀取圖像。請檢查圖像路徑。")
        return

    for i, coord in enumerate(coordinates):
        # 確保每個座標是一對 (x, y) 數值
        x, y = int(coord[0][0]), int(coord[0][1])

        # 標記點
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        # 標記序號
        cv2.putText(image, str(i + 1), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存並顯示標記後的圖像
    cv2.imwrite('marked_image.jpg', image)
    cv2.imshow('Marked Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主程序
if __name__ == "__main__":
    # 假設這是你的座標列表
    xy_list =search.search()
    mark_coordinates('./photo/output_image.jpg', xy_list)  # 將繪製了輪廓的圖像保存為'output_image.jpg'
