import argparse
import json
import logging
import logging.config
import os

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from checker import Checker
from models import SantaCoder

# load logging configuration
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

parser = argparse.ArgumentParser(description="Trained Without My Consent")
parser.add_argument(
    "--language",
    type=str,
    default="py",
    help="language of the code",
)  # programming language
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data",
    help="path to the dataset",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="batch size",
)  # batch size
parser.add_argument(
    "--model",
    type=str,
    default="santa_coder",
    help="model to use",
)
parser.add_argument(
    "--sorted",
    type=bool,
    default=False,
    help="sort the dataset",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="output directory for results (defaults to run_results)",
)
parser.add_argument(
    "--limit_per_class",
    type=int,
    default=50,
    help="maximum number of files to process per class (TheStack and repos)",
)

args = parser.parse_args()

# Determine output directory
if args.output_dir:
    OUTPUT_DIR = args.output_dir
else:
    OUTPUT_DIR = os.path.join(os.getcwd(), "run_results")

print("\033[93m" + f"Output directory: {OUTPUT_DIR}" + "\033[0m")

model = SantaCoder() if args.model == "santa_coder" else None


def get_model_output(file_path) -> int:
    results = []
    file_checker = Checker(file_path)
    model_inputs = [
        file_checker.prepare_inputs_for_infill(level=i)
        for i in [
            "function_names",
            "variable_names",
            "class_names",
            "comments",
            "docstrings",
            "strings",
        ]
    ]
    model_inputs = [input for sublist in model_inputs for input in sublist]

    if model_inputs == []:
        return None
    for candidate_input in tqdm(model_inputs):
        model_output = model.infill(
            (
                candidate_input["infill"],
                candidate_input["prefix"],
                candidate_input["suffix"],
                candidate_input["level"],
            )
        )
        if model_output == "too_many_tokens":
            f = open(os.path.join(OUTPUT_DIR, "too_many_tokens.txt"), "a")
            f.write(file_path + "\n")
            return 400
        else:
            try:
                result = file_checker.check_similarity(
                    model_output,
                    candidate_input,
                    similiarity_metric="exact"
                    if candidate_input["level"]
                    in ["function_names", "variable_names", "class_names"]
                    else "fuzzy",
                )
                results.append(
                    {
                        "file_path": file_path,
                        "level": candidate_input["level"],
                        "similarity_metric": "exact"
                        if candidate_input["level"]
                        in ["function_names", "variable_names", "class_names"]
                        else "fuzzy",
                        "result": result,
                        "similarity_objective": candidate_input["infill"],
                        "model_output": model_output,
                    }
                )
            except Exception as e:
                logging.error(e)
                return 500
    with open(
        os.path.join(OUTPUT_DIR, "results.jsonl"),
        "a",
    ) as f:
        json_results = json.dumps(results)
        f.write(json_results)
        f.write("\n")
    return 200


if __name__ == "__main__":
    print(args.sorted)
    print("Available devices: ", torch.cuda.device_count())

    if torch.cuda.is_available():
        logging.info(f"GPU is available. Running on {torch.cuda.get_device_name(0)}")
    else:
        logging.info("GPU is not available. Running on CPU")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_files = []
    for dirpath, dirnames, filenames in os.walk(args.dataset_path):
        python_files = [file for file in filenames if file.endswith(".py")]
        if python_files:
            dataset_files.extend(
                [os.path.join(dirpath, file) for file in python_files]
            )

    dataset_files = (
        sorted(dataset_files) if args.sorted else sorted(dataset_files, reverse=True)
    )

    # Read already processed files if they exist
    generated_file = os.path.join(OUTPUT_DIR, "generated.txt")
    if os.path.exists(generated_file):
        files_generated_blocks = open(generated_file, "r").readlines()
    else:
        files_generated_blocks = []

    files_generated_blocks = [file.rstrip("\n") for file in files_generated_blocks]

    files_generated_blocks = (
        sorted(files_generated_blocks)
        if args.sorted
        else sorted(files_generated_blocks, reverse=True)
    )

    # Read already processed files if they exist
    processed_tokens_file = os.path.join(OUTPUT_DIR, "processed_tokens.txt")
    if os.path.exists(processed_tokens_file):
        already_processed_files = open(processed_tokens_file, "r").readlines()
        already_processed_files = [file.rstrip("\n") for file in already_processed_files]
    else:
        already_processed_files = []

    # Read dangerous files if they exist
    assert_errors_file = os.path.join(OUTPUT_DIR, "assert_errors.txt")
    if os.path.exists(assert_errors_file):
        dangerous_files = open(assert_errors_file, "r").readlines()
        dangerous_files = [file.rstrip("\n") for file in dangerous_files]
    else:
        dangerous_files = []

    # Read large files if they exist
    too_many_tokens_file = os.path.join(OUTPUT_DIR, "too_many_tokens.txt")
    if os.path.exists(too_many_tokens_file):
        large_files = open(too_many_tokens_file, "r").readlines()
        large_files = [file.rstrip("\n") for file in large_files]
    else:
        large_files = []

    stack_count = 0
    repo_count = 0
    LIMIT = args.limit_per_class
    print(f"Processing up to {LIMIT} files from each class (max {LIMIT * 2} total)")

    for file_path in dataset_files:
        # 1. Check if we should skip this file based on logs
        if file_path in dangerous_files or file_path in large_files:
            print("\033[91m" + file_path + "\033[0m")
            print("Skipping...")
            continue

        # 2. Determine if file is Member (Stack) or Non-Member (Repos)
        is_stack = "the_stack" in file_path

        # 3. Check class-specific limits
        if is_stack and stack_count >= LIMIT:
            continue
        if not is_stack and repo_count >= LIMIT:
            continue
        
        # 4. Process the file
        print("\033[91m" + file_path + "\033[0m")
        print("Processing...")
        
        if file_path in already_processed_files:
            print("Already processed")
            if is_stack:
                stack_count += 1
                print("Stack: " + str(stack_count))
            else:
                repo_count += 1
                print("Repo: " + str(repo_count))
            continue
            
        # Run the model (This function handles writing the results to JSONL)
        output_code = get_model_output(file_path)

        if output_code == 200:
            print("Successfully processed")
            if is_stack:
                stack_count += 1
                print("Stack: " + str(stack_count))
            else:
                repo_count += 1
                print("Repo: " + str(repo_count))
        else:
            print("Error: " + str(output_code))

        # 5. Log that we finished this file
        with open(
            os.path.join(OUTPUT_DIR, "processed_tokens.txt"), "a"
        ) as f:
            f.write(file_path + "\n")

        # 6. Global Stop Condition
        if stack_count >= LIMIT and repo_count >= LIMIT:
            print("Reached limit for both classes. Stopping.")
            break