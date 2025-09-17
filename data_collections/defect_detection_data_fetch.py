import json
import sys
from pathlib import Path
from datasets import load_dataset

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

def fetch_defect_detection_data():
    """Fetch Defect Detection dataset and save filtered data for vulnerable samples only"""

    print("Loading Defect Detection dataset...")

    try:
        # Load the dataset
        dataset = load_dataset("mcanoglu/defect-detection", split="train")
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0

    # Output file
    output_file = config.get_raw_file_path("defect_detection_raw.jsonl")

    print(f"Processing and filtering data...")
    print(f"Output will be saved to: {output_file}")

    processed_count = 0
    vulnerable_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(dataset):
            try:
                processed_count += 1

                # Only keep samples labeled as vulnerable
                label_name = sample.get("label_name", "")
                if label_name != "vulnerable":
                    continue

                vulnerable_count += 1

                # Extract only the required fields
                filtered_entry = {
                    "code": sample.get("code", ""),
                    "description": sample.get("description", "")
                }

                # Skip entries with missing essential data
                if not filtered_entry["code"] or not filtered_entry["description"]:
                    print(f"Skipping sample {i+1} - missing essential fields")
                    vulnerable_count -= 1
                    continue

                # Write to JSONL file
                f.write(json.dumps(filtered_entry, ensure_ascii=False) + '\n')

                # Progress indicator
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count}/{len(dataset)} samples... Found {vulnerable_count} vulnerable")

            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                continue

    print(f"\n✓ Data fetch complete!")
    print(f"Total samples processed: {processed_count}")
    print(f"Vulnerable samples found: {vulnerable_count}")
    print(f"Samples written: {vulnerable_count}")
    print(f"Output saved to: {output_file}")

    return vulnerable_count

def main():
    """Main function to fetch and process Defect Detection data"""

    print("=== Defect Detection Data Fetch ===")
    print("Dataset: mcanoglu/defect-detection")
    print("Filter: label_name = 'vulnerable'")
    print("Fields: code, description")
    print()

    # Fetch and process the data
    count = fetch_defect_detection_data()

    print(f"\n=== Fetch Complete ===")
    print(f"Successfully processed {count} vulnerable entries")

if __name__ == "__main__":
    main()