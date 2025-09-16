import json
import random
import sys
from pathlib import Path

# Set higher limit for large integers in test cases
sys.set_int_max_str_digits(100000)

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

from datasets import load_dataset

def load_seen_names():
    """Load already processed names from taco_raw_seen.jsonl"""
    seen_file = config.get_raw_file_path("taco_raw_seen.jsonl")
    seen_names = set()

    if seen_file.exists():
        with open(seen_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    seen_names.add(data.get("name", ""))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(seen_names)} previously processed names")
    return seen_names

def fetch_taco_data(num_samples=None):
    """
    Fetch samples from TACO dataset and save raw data
    If num_samples is None, process the entire dataset
    """
    print(f"Loading TACO dataset...")

    # Load the dataset
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet",
        split="train",
    )

    print(f"Dataset loaded with {len(ds)} samples")

    # Load previously seen names
    seen_names = load_seen_names()

    # Determine which samples to process
    if num_samples is None:
        indices = range(len(ds))
        print("Processing entire dataset...")
    else:
        indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
        print(f"Processing {len(indices)} random samples...")

    # Prepare output data
    raw_data = []
    skipped_count = 0

    for i, idx in enumerate(indices):
        sample = ds[idx]

        # Get problem name and check if already processed
        problem_name = sample.get("id", f"problem_{idx}")
        if problem_name in seen_names:
            skipped_count += 1
            continue

        # Parse JSON fields
        try:
            solutions = json.loads(sample["solutions"])
            input_output = json.loads(sample["input_output"])
        except json.JSONDecodeError:
            print(f"Skipping sample {idx} due to JSON parsing error")
            continue

        # Extract first solution (usually the best one)
        if solutions and len(solutions) > 0:
            solution_code = solutions[0]
        else:
            print(f"Skipping sample {idx} - no solutions found")
            continue

        # Parse tags and extract relevant metadata
        tags = sample.get("tags", [])
        if isinstance(tags, str):
            # Handle string representation of Python lists
            if tags.startswith('[') and tags.endswith(']'):
                try:
                    # Use eval to parse Python list strings like "['Sorting', 'Data structures']"
                    tags = eval(tags)
                except:
                    # If eval fails, try json.loads
                    try:
                        tags = json.loads(tags)
                    except json.JSONDecodeError:
                        tags = [tags] if tags else []
            else:
                # Single string tag
                tags = [tags] if tags else []

        # Ensure tags is always a list
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []

        difficulty = sample.get("difficulty", "unknown").lower()

        # Determine primary topic from tags
        topic = tags[0] if tags else "general"

        # Get time complexity if available
        time_complexity = sample.get("time_complexity", "unknown")

        # Create raw data entry
        raw_entry = {
            "name": problem_name,
            "question": sample["question"],
            "solution": solution_code,
            "tags": tags,
            "difficulty": difficulty,
            "language": "python",
            "topic": topic,
            "time_complexity": time_complexity
        }

        raw_data.append(raw_entry)
        print(f"Processed sample {i + 1}/{num_samples}")

    # Get output file paths from config
    output_file = config.get_raw_file_path("taco_raw.jsonl")
    seen_file = config.get_raw_file_path("taco_raw_seen.jsonl")

    # Write to JSONL file (append mode to preserve existing data)
    with open(output_file, 'a', encoding='utf-8') as f:
        for entry in raw_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Write to seen file (append mode to track processed names)
    with open(seen_file, 'a', encoding='utf-8') as f:
        for entry in raw_data:
            seen_entry = {"name": entry["name"]}
            f.write(json.dumps(seen_entry, ensure_ascii=False) + '\n')

    total_processed = len(indices)
    print(f"Processed {total_processed} samples:")
    print(f"  - New samples written: {len(raw_data)}")
    print(f"  - Skipped (already seen): {skipped_count}")
    print(f"  - Skipped (other reasons): {total_processed - len(raw_data) - skipped_count}")
    print(f"Output written to: {output_file}")
    return len(raw_data)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Process entire dataset (set to None for full dataset)
    count = fetch_taco_data(num_samples=None)
    print(f"Data collection complete: {count} new samples saved")
