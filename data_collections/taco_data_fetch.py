import json
import random
import sys
from pathlib import Path

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

from datasets import load_dataset

def fetch_taco_data(num_samples=20):
    """
    Fetch random samples from TACO dataset and save raw data
    """
    print(f"Loading TACO dataset...")

    # Load the dataset
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/BAAI/TACO/ALL/train-*.parquet",
        split="train",
    )

    print(f"Dataset loaded with {len(ds)} samples")

    # Get random indices
    random_indices = random.sample(range(len(ds)), min(num_samples, len(ds)))

    # Prepare output data
    raw_data = []

    for i, idx in enumerate(random_indices):
        sample = ds[idx]

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

        # Skip non-Python solutions
        if not solution_code.strip().startswith(('def ', 'class ', 'import ', 'from ')) and 'def ' not in solution_code:
            print(f"Skipping sample {idx} - not a Python solution")
            continue

        # Parse tags and extract relevant metadata
        tags = sample.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = [tags] if tags else []

        difficulty = sample.get("difficulty", "unknown").lower()

        # Determine primary topic from tags
        topic = tags[0] if tags else "general"

        # Create raw data entry
        raw_entry = {
            "question": sample["question"],
            "solution": solution_code,
            "tags": tags,
            "difficulty": difficulty,
            "language": "python",
            "topic": topic
        }

        raw_data.append(raw_entry)
        print(f"Processed sample {i + 1}/{num_samples}")

    # Get output file path from config
    output_file = config.get_raw_file_path("taco_raw.jsonl")

    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in raw_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully wrote {len(raw_data)} samples to {output_file}")
    return len(raw_data)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Fetch 20 random samples
    count = fetch_taco_data(num_samples=20)
    print(f"Data collection complete: {count} samples saved")
