import os
from datetime import datetime
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "./data/unseen"
NUM_SAMPLES = 10
DATASET_NAME = "AISE-TUDelft/the-heap"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Fetching {NUM_SAMPLES} unseen Python files from {DATASET_NAME}...")
    
    # Load The Heap, specifically targeting the Python subset
    try:
        ds = load_dataset(
            DATASET_NAME, 
            "Python", 
            trust_remote_code=True, 
            streaming=True, 
            split="train"
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    samples = []
    
    # Stream through the dataset and filter out contaminated files
    def filter_non_duplicates(item):
        is_exact_dup = item.get("exact_duplicates_stackv2", True)
        is_near_dup = item.get("near_duplicates_stackv2", True)
        return not is_exact_dup and not is_near_dup
    
    filtered_ds = ds.filter(filter_non_duplicates)
    pbar = tqdm(total=NUM_SAMPLES, desc="Downloading files", unit="file")
    
    for item in filtered_ds:
        samples.append(item)
        pbar.update(1)
        
        if len(samples) >= NUM_SAMPLES:
            break
    
    pbar.close()
            
    print(f"Collected {len(samples)} unseen samples. Saving to Parquet...")
    
    hf_dataset = Dataset.from_list(samples)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"unseen_python_{NUM_SAMPLES}_{timestamp}.parquet"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    hf_dataset.to_parquet(output_path)
    
    print(f"Successfully saved to {output_path}")
    print("\nOn your offline compute nodes, you can load this via:")
    print(f"    from datasets import load_dataset")
    print(f"    ds = load_dataset('parquet', data_files='{output_path}')")
    
    os._exit(0)

if __name__ == "__main__":
    main()