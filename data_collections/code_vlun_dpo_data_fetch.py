import json
import sys
from pathlib import Path
from datasets import load_dataset

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

def remove_code_fences(text):
    """Remove code fences (```) from text content"""
    if not text:
        return text

    # Remove triple backticks with optional language specifiers
    import re
    # Remove opening fences like ```python, ```c++, ```
    text = re.sub(r'^```[a-zA-Z0-9]*\n?', '', text, flags=re.MULTILINE)
    # Remove closing fences
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    # Remove any remaining standalone ```
    text = re.sub(r'^```$', '', text, flags=re.MULTILINE)

    return text.strip()

def fetch_code_vulnerability_data():
    """Fetch Code Vulnerability Security DPO dataset and save filtered data"""

    print("Loading Code Vulnerability Security DPO dataset...")

    try:
        # Load the dataset
        dataset = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0

    # Output file
    output_file = config.get_raw_file_path("code_vlun_dpo_raw.jsonl")

    print(f"Processing and filtering data...")
    print(f"Output will be saved to: {output_file}")

    processed_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(dataset):
            try:
                # Extract and clean the required fields
                filtered_entry = {
                    "lang": sample.get("lang", ""),
                    "vulnerability": sample.get("vulnerability", ""),
                    "chosen": remove_code_fences(sample.get("chosen", "")),
                    "rejected": remove_code_fences(sample.get("rejected", ""))
                }

                # Skip entries with missing essential data
                if not filtered_entry["vulnerability"] or not filtered_entry["chosen"] or not filtered_entry["rejected"]:
                    print(f"Skipping sample {i+1} - missing essential fields")
                    continue

                # Write to JSONL file
                f.write(json.dumps(filtered_entry, ensure_ascii=False) + '\n')
                processed_count += 1

                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(dataset)} samples...")

            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                continue

    print(f"\n✓ Data fetch complete!")
    print(f"Total samples processed: {len(dataset)}")
    print(f"Samples written: {processed_count}")
    print(f"Output saved to: {output_file}")

    return processed_count

def main():
    """Main function to fetch and process Code Vulnerability DPO data"""

    print("=== Code Vulnerability DPO Data Fetch ===")
    print("Dataset: CyberNative/Code_Vulnerability_Security_DPO")
    print("Fields: lang, vulnerability, chosen, rejected")
    print()

    # Fetch and process the data
    count = fetch_code_vulnerability_data()

    print(f"\n=== Fetch Complete ===")
    print(f"Successfully processed {count} entries")

if __name__ == "__main__":
    main()