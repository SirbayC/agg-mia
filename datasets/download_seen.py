import os
from datasets import load_dataset, Dataset

# Configuration
OUTPUT_DIR = "/scratch/cosminvasilesc/AGG-MIA/datasets/seen"
NUM_SAMPLES = 1000
# StarCoder2-3B was specifically trained on the "smol" subset
DATASET_NAME = "bigcode/the-stack-v2-train-smol" 

def main():
    # Ensure the target scratch directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Fetching {NUM_SAMPLES} Python files from {DATASET_NAME}...")
    print("Ensure you have authenticated via `huggingface-cli login` if needed.")
    
    # We use streaming=True because the dataset is hundreds of GBs.
    # Streaming allows us to download only what we need on the fly.
    try:
        ds = load_dataset(
            DATASET_NAME, 
            trust_remote_code=True, 
            streaming=True, 
            split="train"
        )
    except Exception as e:
        print(f"Error loading dataset. Please check your HF token/access: {e}")
        return

    samples = []
    
    # Iterate through the stream until we find 1000 Python files
    for item in ds:
        # The 'smol' dataset contains 17 languages mixed together. 
        # We check the language metadata flag to isolate Python.
        lang = item.get("language", item.get("lang", ""))
        
        if lang.lower() == "python":
            samples.append(item)
        
        if len(samples) >= NUM_SAMPLES:
            break
            
    print(f"Collected {len(samples)} samples. Saving to Parquet...")
    
    # Convert the list of dicts to a HF Dataset and save it
    hf_dataset = Dataset.from_list(samples)
    output_path = os.path.join(OUTPUT_DIR, "seen_python_1000.parquet")
    hf_dataset.to_parquet(output_path)
    
    print(f"Successfully saved to {output_path}")
    print("\nOn your offline compute nodes, you can load this via:")
    print(f"    from datasets import load_dataset")
    print(f"    ds = load_dataset('parquet', data_files='{output_path}')")

if __name__ == "__main__":
    main()