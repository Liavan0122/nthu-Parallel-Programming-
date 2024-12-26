import random

# 定義矩陣大小
rows, cols = 10000, 10000

# 初始化矩陣為隨機數據 (0, 1, 2)
grid = [[1 for _ in range(cols)] for _ in range(rows)]

grid[0][0] = 2

# 構造輸出格式
output = f"{rows} {cols} {grid}"

# 將測資保存到 .txt 文件
with open("06huge.txt", "w") as file:
    file.write(output)

print("測資已成功保存到 'rotting_oranges_test.txt'")
