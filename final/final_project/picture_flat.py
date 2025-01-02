import cv2
import numpy as np

def perform_perspective_transform(input_image_path, output_image_path='output_transformed_image.jpg'):
    """
    進行透視變換，並顯示及保存結果。

    :param input_image_path: 輸入圖像的路徑
    :param output_image_path: 變換後的圖像保存路徑
    :return: 透視變換後的圖像
    """
    points = []  # 用於存儲用戶選擇的四個點

    # 鼠標點擊回調函數
    def click_and_select(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # 在圖像上繪製一個圓圈標記
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow('Image', image)
            if len(points) == 4:  # 當選擇了四個點時停止
                cv2.waitKey(0)

    # 讀取圖像
    image = cv2.imread(input_image_path)
    
    # 檢查圖像是否成功加載
    if image is None:
        print("無法加載圖像，請檢查圖像路徑。")
        return None

    # 顯示圖像，並等待鼠標點擊
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', click_and_select)

    # 等待用戶選擇四個點
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 確保選擇了四個點
    if len(points) == 4:
        # 取得用戶選擇的四個點
        pts1 = np.float32(points)
        
        # 定義目標圖像中的四個角點為原圖的大小
        height, width = image.shape[:2]
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        # 計算透視變換矩陣
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 應用透視變換
        result = cv2.warpPerspective(image, matrix, (width, height))
        
        # 顯示透視變換後的圖像
        cv2.imshow('Warped Image', result)
        cv2.waitKey(0)
        
        # 保存輸出的變換圖像
        cv2.imwrite(output_image_path, result)
        print(f"透視變換後的圖像已保存至: {output_image_path}")
        
        cv2.destroyAllWindows()
        
        return result
    else:
        print("未選擇足夠的點。請選擇四個點來進行透視變換。")
        return None


# 範例使用：
input_path = './photo/flat.png'  # 輸入圖像路徑
output_path = './photo/output_transformed_image.jpg'  # 輸出圖像保存路徑

# 調用函數進行透視變換
perform_perspective_transform(input_path, output_path)
