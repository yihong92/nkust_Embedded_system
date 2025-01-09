def calculate_total_price(all_line, price_list):
    """
    計算每項價錢的總金額

    :param all_line: 每個格子中的線條數量列表
    :param price_list: 每個格子的單價列表
    :return: 計算的總金額
    """
    if len(all_line) != len(price_list):
        raise ValueError("all_line 和 price_list 的長度必須相同")

    # 相乘計算每項價錢
    result = [a * b for a, b in zip(all_line, price_list)]

    # 每項相加得到總金額
    total = sum(result)
    return total
