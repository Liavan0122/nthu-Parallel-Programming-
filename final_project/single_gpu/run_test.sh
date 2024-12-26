#!/bin/bash

# Compile the program
make

# Run small all test cases  1 <= m, n <= 10 , probability [random]
for i in {1..5}; do
    input_file="../testcases/0${i}.txt"
    output_file="./answer/0${i}_ans.txt"
    expected_file="../testcases/0${i}_answer.txt"

    srun -N1 -n1 --gres=gpu:1 ./orange_single_gpu "${input_file}" "${output_file}"

    # Compare the output with the expected answer using a custom compare program
    if ./compare_output "${output_file}" "${expected_file}"; then
        echo "Test case ${i} passed."
    else
        echo "Test case ${i} failed. Output does not match expected result."
    fi
done

# Run large all test cases 20 <= m, n <= 50 ,probability [0 : 1 : 2] [15, 75, 10]

for i in {1..5}; do
    input_file="../testcases/0${i}large.txt"
    output_file="./answer/0${i}large_ans.txt"
    expected_file="../testcases/0${i}large_answer.txt"

    srun -N1 -n1 --gres=gpu:1 ./orange_single_gpu "${input_file}" "${output_file}"

    # Compare the output with the expected answer using a custom compare program
    if ./compare_output "${output_file}" "${expected_file}"; then
        echo "Test large case ${i} passed."
    else
        echo "Test large case ${i} failed. Output does not match expected result."
    fi
done

# # Run large all test cases 500 <= m, n <= 100  ,probability [0 : 1 : 2] [5, 90, 5]

for i in {1..5}; do
    input_file="../testcases/0${i}huge.txt"
    output_file="./answer/0${i}huge_ans.txt"
    expected_file="../testcases/0${i}huge_answer.txt"

    srun -N1 -n1 --gres=gpu:1 ./orange_single_gpu "${input_file}" "${output_file}"

    # Compare the output with the expected answer using a custom compare program
    if ./compare_output "${output_file}" "${expected_file}"; then
        echo "Test huge case ${i} passed."
    else
        echo "Test huge case ${i} failed. Output does not match expected result."
    fi
done

# delete the program
make clean