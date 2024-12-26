import os
import json
import csv

def benchmark_results_to_csv(directory, output_csv):
    # Define the header for the CSV file based on the expected JSON structure
    header = [
        "Batch Size", "Seq Length", "Num Heads", "Emb Dim", "Implementation", "Causal",
        "Forward Time (s)", "Forward FLOPS (TFLOPs/s)",
        "Backward Time (s)", "Backward FLOPS (TFLOPs/s)",
        "Forward_Backward Time (s)", "Forward_Backward FLOPS (TFLOPs/s)",
        "Peak Memory Usage (MB)", "Status"
    ]

    rows = []

    # Iterate through all possible parameter combinations
    batch_sizes = [16, 32, 64]
    seq_lengths = [512, 1024, 2048]
    num_heads_list = [8, 16, 32]
    emb_dims = [512, 1024, 2048]
    implementations = ["Pytorch", "Flash2"]
    causal_flags = [True, False]

    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            for num_heads in num_heads_list:
                for emb_dim in emb_dims:
                    for implementation in implementations:
                        for causal in causal_flags:
                            # Construct the expected filename
                            causal_str = "true" if causal else "false"
                            filename = f"bs{batch_size}_seq{seq_length}_heads{num_heads}_emb{emb_dim}_impl{implementation}_causal{causal_str}.json"
                            file_path = os.path.join(directory, filename)

                            if os.path.exists(file_path):
                                # Load the JSON content
                                with open(file_path, 'r') as json_file:
                                    data = json.load(json_file)

                                    # Extract values for each field in the JSON file
                                    forward_time = data["forward"].get("time(s)", None)
                                    forward_flops = data["forward"].get("FLOPS(TFLOPs/s)", None)
                                    backward_time = data["backward"].get("time(s)", None)
                                    backward_flops = data["backward"].get("FLOPS(TFLOPs/s)", None)
                                    forward_backward_time = data.get("forward_backward", {}).get("time(s)", None)
                                    forward_backward_flops = data.get("forward_backward", {}).get("FLOPS(TFLOPs/s)", None)
                                    peak_memory_usage = data.get("peak_memory_usage(MB)", None)

                                    # Append the row data with status "Success"
                                    rows.append([
                                        batch_size, seq_length, num_heads, emb_dim, implementation, causal,
                                        forward_time, forward_flops,
                                        backward_time, backward_flops,
                                        forward_backward_time, forward_backward_flops,
                                        peak_memory_usage, "Success"
                                    ])
                            else:
                                # Append the row data with status "Out of Memory"
                                rows.append([
                                    batch_size, seq_length, num_heads, emb_dim, implementation, causal,
                                    None, None, None, None, None, None, None, "Out of Memory"
                                ])

    # Write data to CSV file
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Benchmark results have been saved to {output_csv}")

# Example usage
directory_path = "benchmark_results"
output_csv_path = "benchmark_results_summary.csv"
benchmark_results_to_csv(directory_path, output_csv_path)
