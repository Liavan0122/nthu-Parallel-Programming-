#!/bin/bash

# # 編譯程式
gcc -pthread -o rotting_oranges rotting_oranges.cc
if [ $? -ne 0 ]; then
    echo "編譯失敗！請檢查程式代碼。"
    exit 1
fi

# 設定測試資料與答案目錄
testcase_dir="../testcases"
output_dir="./outputs"
mkdir -p $output_dir

# 初始化計數器
pass_count=0
fail_count=0

# 遍歷所有測試資料
for testcase in "$testcase_dir"/*.txt; do
    # 忽略 _answer.txt 檔案
    if [[ "$testcase" == *_answer.txt ]]; then
        continue
    fi

    # 取得檔名（不包含路徑與副檔名）
    base_name=$(basename "$testcase" .txt)

    # 開始計時（納秒）
    start_time=$(date +%s%N)

    # 執行程式並將輸出存入臨時檔案
    ./rotting_oranges "$testcase" "$output_dir/${base_name}_out.txt"
    # ./orange_seq "$testcase" "$output_dir/${base_name}_out.txt"

    # 結束計時（納秒）
    end_time=$(date +%s%N)

    # 計算執行時間（以秒為單位）
    elapsed_time=$(awk "BEGIN {printf \"%.6f\", ($end_time - $start_time) / 1000000000}")

    # 比較輸出與正確答案
    if cmp -s <(sed '/^$/d' "$output_dir/${base_name}_out.txt" | tr -d '[:space:]') <(sed '/^$/d' "$testcase_dir/${base_name}_answer.txt" | tr -d '[:space:]'); then
        echo "Test case $base_name: PASS (Time: ${elapsed_time}s)"
        pass_count=$((pass_count + 1))
    else
        echo "Test case $base_name: FAIL (Time: ${elapsed_time}s)"
        echo "錯誤輸出："
        echo "--- 程式輸出 ---"
        cat "$output_dir/${base_name}_out.txt"
        echo "--- 正確答案 ---"
        cat "$testcase_dir/${base_name}_answer.txt"
        echo "---------------"
        fail_count=$((fail_count + 1))
    fi
done

# 顯示總結
echo "==========================="
echo "Total PASS: $pass_count"
echo "Total FAIL: $fail_count"
echo "==========================="

# 強制退出腳本
echo "腳本執行完畢，退出中..."
exit 0
