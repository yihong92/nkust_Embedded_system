import numpy as np

def calculate_1_norm_distance(hist1, hist2,th):
    """
    計算兩個直方圖之間的 1-范數距離 (L1 距離)
    
    參數:
    hist1 (numpy.ndarray): 第一個直方圖
    hist2 (numpy.ndarray): 第二個直方圖
    
    返回:
    float: 1-范數距離
    """

    distance = np.sum(np.abs(hist1 - hist2))
    if distance<=th:
        return 1
    else:
        return 0 
    

