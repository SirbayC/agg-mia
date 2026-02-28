import os
import time
import requests
from datetime import datetime
from datasets import load_dataset, Dataset

OUTPUT_DIR = "./data/seen"
NUM_SAMPLES = 10
DATASET_NAME = "bigcode/the-stack-v2-train-smol-ids"

SWH_TOKEN = os.environ.get("SWH_TOKEN")


def is_python_file(file_obj):
    language = str(file_obj.get("language", file_obj.get("lang", ""))).lower()
    if language == "python":
        return True

    path = str(file_obj.get("path", file_obj.get("filename", ""))).lower()
    return path.endswith(".py")

def fetch_file_content(file_obj, token=None):
    """Fetches raw file content from Software Heritage using the public API."""
    blob_id = file_obj.get("blob_id")
    if not blob_id:
        return None

    url = f"https://archive.softwareheritage.org/api/1/content/sha1:{blob_id}/raw/"
    src_encoding = file_obj.get("src_encoding") or "utf-8"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    while True:
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                out = dict(file_obj)
                out["content"] = response.content.decode(src_encoding, errors="replace")
                return out
            
            elif response.status_code == 429:
                print("Rate limited by Software Heritage. Sleeping for 10 seconds...")
                time.sleep(10)
                continue # Retry the request
                
            else:
                out = dict(file_obj)
                out["status_code"] = f"HTTP {response.status_code}"
                return out
                
        except Exception as e:
            out = dict(file_obj)
            out["status_code"] = str(e)
            return out

def main():
    # Ensure the target scratch directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Fetching {NUM_SAMPLES} Python files from {DATASET_NAME} via SWH API...")
    print("Ensure you have authenticated via `hf auth login` to access the metadata.")
    
    # We use streaming=True because the dataset is hundreds of GBs.
    try:
        ds = load_dataset(
            DATASET_NAME, 
            streaming=True,
            split="train",
        )
    except Exception as e:
        print(f"Error loading dataset. Please check your HF token/access: {e}")
        return

    samples = []
    
    # Iterate through stream rows and collect Python files with retrieved content.
    for row in ds:
        files = row.get("files", [])
        if not isinstance(files, list):
            continue

        for file_obj in files:
            if not isinstance(file_obj, dict):
                continue
            if not is_python_file(file_obj):
                continue

            # Fetch the actual code from SWH
            enriched = fetch_file_content(file_obj,token=SWH_TOKEN)
            if enriched is None or "status_code" in enriched:
                # Skip files we couldn't download successfully
                continue

            samples.append(enriched)
            print(f"Successfully downloaded {len(samples)}/{NUM_SAMPLES} files...", end="\r")
            
            # Sleep slightly to avoid hammering the API too hard
            time.sleep(0.5)
            
            if len(samples) >= NUM_SAMPLES:
                break

        if len(samples) >= NUM_SAMPLES:
            print("\nFinished downloading files.")
            break
            
    print(f"Collected {len(samples)} samples. Saving to Parquet...")
    
    # Convert the list of dicts to a HF Dataset and save it
    hf_dataset = Dataset.from_list(samples)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"seen_python_{NUM_SAMPLES}_{timestamp}.parquet"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    hf_dataset.to_parquet(output_path)
    
    print(f"Successfully saved to {output_path}")
    print("\nOn your offline compute nodes, you can load this via:")
    print(f"    from datasets import load_dataset")
    print(f"    ds = load_dataset('parquet', data_files='{output_path}')")

if __name__ == "__main__":
    main()