import argparse
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Display sample files from a parquet dataset.")
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="./data/seen/seen_python_100.parquet",
        help="Path to the parquet file.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Number of files to display.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1000,
        help="How many content characters to print per file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ds = load_dataset("parquet", data_files=args.parquet_path, split="train")

    total = len(ds)
    print(f"Loaded {total} rows from {args.parquet_path}")
    print("=" * 80)

    shown = min(args.num_files, total)
    for i in range(shown):
        row = ds[i]
        path = row.get("path") or row.get("filename") or "<unknown_path>"
        blob_id = row.get("blob_id", "<no_blob_id>")
        content = row.get("content", "")
        preview = content[: args.preview_chars]

        print(f"[{i + 1}/{shown}] {path}")
        print(f"blob_id: {blob_id}")
        print("--- content preview ---")
        print(preview)
        print("=" * 80)


if __name__ == "__main__":
    main()
