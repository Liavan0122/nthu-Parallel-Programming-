#! /bin/bash
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --output=final.log
#SBATCH --error=final.err
module purge
module load mpi
# Compile the program
make
NPROCESS=4
TIMESTAMP_LOG="./timelog_hybrid2.txt"
script_method="hybrid2"
: > "${TIMESTAMP_LOG}"
echo "Execution started at: $(date)" > "${TIMESTAMP_LOG}"
# Run small all test cases  1 <= m, n <= 10 , probability [random] 
#Run large all test cases 500 <= m, n <= 100  ,probability [0 : 1 : 2] [5, 90, 5]
# Run large all test cases 20 <= m, n <= 50 ,probability [0 : 1 : 2] [15, 75, 10]

for i in {1..5}; do
    input_file="../testcases/0${i}.txt"
    output_file="./answer/0${i}_ans.txt"
    expected_file="../testcases/0${i}_answer.txt"
    start_time=$(date +%s.%N)
    mpirun -n ${NPROCESS} ./orange_${script_method} "${input_file}" "${output_file}"
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)  
    formatted_time=$(printf "%.6f" "${elapsed_time}")
    echo "Small test case ${i}: ${formatted_time}s" >> "${TIMESTAMP_LOG}"
    ### Compare the output with the expected answer using a custom compare program
    if ./compare_output "${output_file}" "${expected_file}"; then
        echo "Test case ${i} passed.">> "${TIMESTAMP_LOG}"
    else
        echo "Test case ${i} failed. Output does not match expected result.">> "${TIMESTAMP_LOG}"
    fi
done

for i in {1..5}; do
    input_file="../testcases/0${i}large.txt"
    output_file="./answer/0${i}large_ans.txt"
    expected_file="../testcases/0${i}large_answer.txt"
    start_time=$(date +%s.%N)
    mpirun -n ${NPROCESS} ./orange_${script_method} "${input_file}" "${output_file}"
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    formatted_time=$(printf "%.6f" "${elapsed_time}")
    echo "Large test case ${i}: ${formatted_time}s" >> "${TIMESTAMP_LOG}"
    # Compare the output with the expected answer using a custom compare program
    if ./compare_output "${output_file}" "${expected_file}"; then
        echo "Test case ${i} passed.">> "${TIMESTAMP_LOG}"
    else
        echo "Test case ${i} failed. Output does not match expected result.">> "${TIMESTAMP_LOG}"
    fi
done

for i in {1..5}; do
    input_file="../testcases/0${i}huge.txt"
    output_file="./answer/0${i}huge_ans.txt"
    expected_file="../testcases/0${i}huge_answer.txt"
    start_time=$(date +%s.%N)
    mpirun -n ${NPROCESS} ./orange_${script_method} "${input_file}" "${output_file}"
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    formatted_time=$(printf "%.6f" "${elapsed_time}")
    echo "Huge test case ${i}: ${formatted_time}s" >> "${TIMESTAMP_LOG}"
    # Compare the output with the expected answer using a custom compare program
    if ./compare_output "${output_file}" "${expected_file}"; then
        echo "Test case ${i} passed.">> "${TIMESTAMP_LOG}"
    else
        echo "Test case ${i} failed. Output does not match expected result.">> "${TIMESTAMP_LOG}"
    fi
done

# for i in {1..1}; do
#     input_file="../testcases/0${i}very_extra_huge.txt"
#     output_file="./answer/0${i}very_extra_huge_ans.txt"
#     expected_file="../testcases/0${i}very_extra_huge.txt"
#     start_time=$(date +%s.%N)
#     mpirun -n ${NPROCESS} ./orange_${script_method} "${input_file}" "${output_file}"
#     end_time=$(date +%s.%N)
#     elapsed_time=$(echo "$end_time - $start_time" | bc)
#     echo "very_extra_huge test case ${i}: ${elapsed_time}s" >> "${TIMESTAMP_LOG}"
#     # Compare the output with the expected answer using a custom compare program
#     if ./compare_output "${output_file}" "${expected_file}"; then
#         echo "Test case ${i} passed.">> "${TIMESTAMP_LOG}"
#     else
#         echo "Test case ${i} failed. Output does not match expected result.">> "${TIMESTAMP_LOG}"
#     fi
# done






# delete the program
make clean
