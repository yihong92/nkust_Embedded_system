import cv2
import numpy as np

def flatten(img):
    # 將圖像轉為灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 Canny 檢測邊緣
    edges = cv2.Canny(gray, 50, 150)

    # 找到輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 假設最大輪廓是目標紙張
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # 找到矩形
            target = approx
            break

    # 定義透視變換的目標大小
    height = 680  # 拉平後的高度
    width = 1440   # 拉平後的寬度
    pts1 = np.float32(target)  # 原始四個點
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # 透視變換
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (width, height))

    # 顯示結果
    cv2.imshow("Warped Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

