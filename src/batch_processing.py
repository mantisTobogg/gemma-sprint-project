# 3. batch_processing.py – Batch Processing Logic

# 	•	Purpose: Process data (comments) in chunks to avoid memory issues.
# 	•	Equivalent Cells in Original .ipynb:
# 	•	No specific cell, but it addresses performance bottlenecks when processing large datasets (improves overall pipeline efficiency).



# batch_processing.py – Batch Processing Logic

def batch_process(data, batch_size, process_func):
    """Process data in batches to optimize memory usage."""
    for i in range(0, len(data), batch_size):
        yield process_func(data[i:i + batch_size])