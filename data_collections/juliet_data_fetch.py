import json
import sys
import random
from pathlib import Path
from datasets import load_dataset

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

def fetch_juliet_data():
    """Fetch Juliet Test Suite dataset with sampling from each class"""

    print("Loading Juliet Test Suite dataset...")

    try:
        # Load the dataset
        dataset = load_dataset("LorenzH/juliet_test_suite_c_1_3", split="train")
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0

    # Output file
    output_file = config.get_raw_file_path("juliet_raw.jsonl")

    print(f"Processing and sampling data...")
    print(f"Output will be saved to: {output_file}")

    # Group samples by class
    class_samples = {}
    total_processed = 0

    for i, sample in enumerate(dataset):
        try:
            class_key = sample.get("class", None)

            # Skip samples without class
            if class_key is None:
                continue

            # Initialize class list if not exists
            if class_key not in class_samples:
                class_samples[class_key] = []

            # Extract only good and bad data, discard others
            good_data = sample.get("good", "")
            bad_data = sample.get("bad", "")

            if good_data or bad_data:
                class_samples[class_key].append({
                    "good": good_data,
                    "bad": bad_data
                })

            total_processed += 1

            # Progress indicator
            if total_processed % 5000 == 0:
                print(f"Processed {total_processed} samples... Found {len(class_samples)} classes")

        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            continue

    print(f"\nFound {len(class_samples)} classes")
    print(f"Classes range: {min(class_samples.keys()) if class_samples else 'N/A'} to {max(class_samples.keys()) if class_samples else 'N/A'}")

    # Sample up to 25 items from each class
    sampled_data = []
    samples_per_class = 25

    for class_key, samples in class_samples.items():
        # Randomly sample up to 25 items from this class
        if len(samples) > samples_per_class:
            selected_samples = random.sample(samples, samples_per_class)
        else:
            selected_samples = samples

        # Add class information to each sample
        for sample in selected_samples:
            sample['class'] = class_key
            sampled_data.append(sample)

        print(f"Class {class_key}: Selected {len(selected_samples)} from {len(samples)} samples")

    # Shuffle the final data
    random.shuffle(sampled_data)

    # Write to JSONL file
    written_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in sampled_data:
            # Skip entries with both good and bad empty
            if not entry["good"] and not entry["bad"]:
                continue

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            written_count += 1

    print(f"\n✓ Data fetch complete!")
    print(f"Total samples processed: {total_processed}")
    print(f"Classes found: {len(class_samples)}")
    print(f"Samples written: {written_count}")
    print(f"Output saved to: {output_file}")

    return written_count

def main():
    """Main function to fetch and process Juliet Test Suite data"""

    print("=== Juliet Test Suite Data Fetch ===")
    print("Dataset: LorenzH/juliet_test_suite_c_1_3")
    print("Sampling: Up to 25 samples per class")
    print("Fields: good, bad, class")
    print()

    # Set random seed for reproducibility
    random.seed(42)

    # Fetch and process the data
    count = fetch_juliet_data()

    print(f"\n=== Fetch Complete ===")
    print(f"Successfully processed {count} entries")

if __name__ == "__main__":
    main()