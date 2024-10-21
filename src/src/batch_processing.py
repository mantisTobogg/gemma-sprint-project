# batch_processing.py
import torch
import logging
from config import CONFIG  # Ensure CONFIG is correctly imported
import time

def batch_process(data, batch_size, process_func):
    """Process data in batches using the specified function."""
    if not data:
        logging.warning("No data provided for batch processing.")
        return

    logging.info("Starting batch processing...")
    total_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(0, len(data), batch_size):
        start_time = time.time()
        batch = data[i:i + batch_size]
        batch_number = i // batch_size + 1
        logging.info(f"Processing batch {batch_number}/{total_batches}...")

        try:
            # Adjust batch processing logic to match function requirements
            if isinstance(batch[0], str):
                # Text batch processing
                result = [process_func(comment) for comment in batch]
                yield result
            else:
                # Tensor batch processing
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(CONFIG["device"])
                yield process_func(batch_tensor)
        except Exception as e:
            logging.error(f"Error processing batch {batch_number}: {e}")
            continue

        end_time = time.time()
        logging.info(f"Batch {batch_number} processed in {end_time - start_time:.2f} seconds")

    logging.info("Batch processing completed.")