#!/bin/bash

# Create an output directory for JSON files
output_dir="benchmark_results"
mkdir -p $output_dir

# Define parameter ranges
batch_sizes=(16 32 64)
seq_lengths=(512 1024 2048)
num_heads=(8 16 32)
emb_dims=(512 1024 2048)
implementations=("Pytorch" "Flash2")
causal_flags=(true false)
repeats=30  # Fixed number of repeats

# Iterate through all parameter combinations
for batch_size in "${batch_sizes[@]}"; do
  for seq_len in "${seq_lengths[@]}"; do
    for num_heads in "${num_heads[@]}"; do
      for emb_dim in "${emb_dims[@]}"; do
        for impl in "${implementations[@]}"; do
          for causal in "${causal_flags[@]}"; do
            # Generate output filename
            output_file="$output_dir/bs${batch_size}_seq${seq_len}_heads${num_heads}_emb${emb_dim}_impl${impl}_causal${causal}.json"
            
            # Execute the script with current parameters
            echo "Running test: batch_size=$batch_size, seq_len=$seq_len, num_heads=$num_heads, emb_dim=$emb_dim, impl=$impl, causal=$causal"
            
            python lab5.py \
              --batch_size $batch_size \
              --seq_len $seq_len \
              --num_heads $num_heads \
              --emb_dim $emb_dim \
              --impl $impl \
              $( [ "$causal" = true ] && echo "--causal" ) \
              --repeats $repeats \
              --output $output_file
          done
        done
      done
    done
  done
done
