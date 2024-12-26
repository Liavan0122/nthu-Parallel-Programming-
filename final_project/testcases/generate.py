import os
import random

# Generate new test cases with adjusted probabilities
def generate_test_case(m, n):
    grid = [[0] * n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            grid[i][j] = random.choices([0, 1, 2], weights=[5, 90, 5])[0]
    
    # Ensure at least one fresh orange (1) and one rotten orange (2)
    grid[random.randint(0, m - 1)][random.randint(0, n - 1)] = 2
    grid[random.randint(0, m - 1)][random.randint(0, n - 1)] = 1
    
    return grid

def calculate_answer(grid):
    from collections import deque

    m, n = len(grid), len(grid[0])
    queue = deque()
    fresh_oranges = 0

    # Initialize the queue with all rotten oranges and count fresh oranges
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                queue.append((i, j, 0))  # (x, y, time)
            elif grid[i][j] == 1:
                fresh_oranges += 1

    if fresh_oranges == 0:
        return 0  # No fresh oranges, answer is 0

    # Perform BFS to rot all reachable fresh oranges
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    time = 0

    while queue:
        x, y, time = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                grid[nx][ny] = 2  # Rot the orange
                queue.append((nx, ny, time + 1))
                fresh_oranges -= 1

    return time if fresh_oranges == 0 else -1  # Return -1 if not all oranges can rot

# Create test cases with adjusted probabilities and correct answers
new_test_cases = []
for i in range(1, 6):  # Use `i` to track test case index
    while True:  # Keep generating until we get a valid answer
        m, n = random.randint(500, 1000), random.randint(500, 1000)
        grid = generate_test_case(m, n)
        answer = calculate_answer([row[:] for row in grid])  # Copy grid to avoid modification
        if answer != -1:  # Only accept cases where answer is not -1
            new_test_cases.append({"m": m, "n": n, "grid": grid, "answer": answer})
            # Terminal output
            print(f"Generate {i:02d}huge.txt and {i:02d}huge_answer.txt")
            break

# Define output directory
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# Write test cases and answers to separate files
for i, test_case in enumerate(new_test_cases, start=1):
    # Write test case
    test_case_path = os.path.join(output_dir, f"{i:02d}huge.txt")
    with open(test_case_path, "w") as f:
        f.write(f"{test_case['m']} {test_case['n']} {test_case['grid']}")
    
    # Write answer
    answer_path = os.path.join(output_dir, f"{i:02d}huge_answer.txt")
    with open(answer_path, "w") as f:
        f.write(str(test_case["answer"]))



output_dir
