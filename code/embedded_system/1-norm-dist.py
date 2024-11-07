import numpy as np

def one_norm_distance(v1, v2):
    """
    計算兩個向量之間的1-范數距離（曼哈頓距離）。

    :param v1: 第一個向量（NumPy數組）
    :param v2: 第二個向量（NumPy數組）
    :return: 1-范數距離（曼哈頓距離）
    """
    # 確保兩個向量的形狀一致
    if v1.shape != v2.shape:
        raise ValueError("向量的大小不一致！")
    
    # 計算元素差的絕對值，然後對所有元素求和
    distance = np.sum(np.abs(v1 - v2))
    return distance

# 測試示例
v1 = np.array([1, 2, 3, 4])
v2 = np.array([4, 5, 6, 7])

distance = one_norm_distance(v1, v2)
print(f"1-范數距離（曼哈頓距離）: {distance}")