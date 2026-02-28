import argparse
import os
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Display sample files from a parquet dataset.")
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=None,
        help="Path to the parquet file. If not specified, searches ./data/seen/ and ./data/unseen/ for the first parquet file.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=1,
        help="Number of files to display.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=None,
        help="How many content characters to print per file. If not specified, shows complete file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Find parquet files to display
    parquet_paths = []
    
    if args.parquet_path:
        # User specified a path
        parquet_paths = [args.parquet_path]
    else:
        # Search in both seen and unseen directories and take only the first from each
        for data_dir in ["./data/seen", "./data/unseen"]:
            if os.path.exists(data_dir):
                parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
                if parquet_files:
                    parquet_paths.append(os.path.join(data_dir, parquet_files[0]))
    
    if not parquet_paths:
        print("Error: No parquet files found in ./data/seen/ or ./data/unseen/")
        return

    # Display samples from each parquet file
    for parquet_path in parquet_paths:
        print(f"\n{'='*80}")
        print(f"Loading: {parquet_path}")
        print(f"{'='*80}")
        
        ds = load_dataset("parquet", data_files=parquet_path, split="train")

        total = len(ds)
        
        # Count rows with non-empty content
        non_empty_count = sum(1 for row in ds if row.get("content", "").strip())
        
        print(f"Loaded {total} rows from {parquet_path}")
        print(f"Rows with non-empty content: {non_empty_count}/{total}")
        print("=" * 80)

        shown = min(args.num_files, total)
        for i in range(shown):
            row = ds[i]
            path = row.get("path") or row.get("filename") or "<unknown_path>"
            blob_id = row.get("blob_id", "<no_blob_id>")
            content = row.get("content", "")
            
            # Show complete file if preview_chars is None
            if args.preview_chars is None:
                preview = content
            else:
                preview = content[: args.preview_chars]

            print(f"[{i + 1}/{shown}] {path}")
            print(f"blob_id: {blob_id}")
            print(f"Content length: {len(content)} characters")
            print("--- content ---")
            print(preview)
            print("=" * 80)


if __name__ == "__main__":
    main()
