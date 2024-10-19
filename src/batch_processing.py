# batch_processing.py

import torch
import logging
from config import CONFIG  # Import CONFIG from the config module

def batch_process(data, batch_size, process_func):
    """Process data in batches using the specified function."""
    logging.info("Starting batch processing...")
    total_batches = (len(data) + batch_size - 1) // batch_size

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_number = i // batch_size + 1
        logging.info(f"Processing batch {batch_number}/{total_batches}...")

        try:
            if isinstance(batch[0], str):
                # Text batch processing
                result = [process_func(comment) for comment in batch]
                yield torch.tensor(result, dtype=torch.float32).to(CONFIG["device"])
            else:
                # Tensor batch processing
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(CONFIG["device"])
                yield process_func(batch_tensor)
        except ValueError as e:
            logging.error(f"Error processing batch {batch_number}: {e}")
            continue

    logging.info("Batch processing completed.")
