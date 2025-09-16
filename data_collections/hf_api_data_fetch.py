import json
import sys
from pathlib import Path

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

def find_all_jsonl_files(hf_dir):
    """Find all JSONL files in the hf directory"""
    hf_path = Path(hf_dir)

    if not hf_path.exists():
        print(f"HF directory not found: {hf_path}")
        return []

    jsonl_files = list(hf_path.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {hf_path}")

    return jsonl_files

def combine_jsonl_files(jsonl_files, output_file):
    """Combine all JSONL files into one output file with standardized format"""
    if not jsonl_files:
        print("No JSONL files to combine")
        return 0

    total_lines = 0

    print(f"Combining JSONL files into: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for jsonl_file in jsonl_files:
            print(f"Processing: {jsonl_file}")

            try:
                with open(jsonl_file, 'r', encoding='utf-8') as infile:
                    lines_in_file = 0
                    for line in infile:
                        line = line.strip()
                        if line:  # Skip empty lines
                            # Validate and parse JSON
                            try:
                                data = json.loads(line)

                                # Transform to standardized format
                                total_lines += 1
                                raw_entry = {
                                    "name": f"hf_api_{total_lines}",
                                    "question": data.get("question", ""),
                                    "solution": data.get("solution", ""),
                                    "tags": ["AI", "LLM", "huggingface", "API"],
                                    "difficulty": "unknown",
                                    "language": "python",
                                    "topic": ["AI", "LLM", "huggingface", "API"],
                                    "time_complexity": "unknown"
                                }

                                outfile.write(json.dumps(raw_entry, ensure_ascii=False) + '\n')
                                lines_in_file += 1

                            except json.JSONDecodeError as e:
                                print(f"Skipping invalid JSON line in {jsonl_file}: {e}")
                                continue

                    print(f"  Added {lines_in_file} lines from {jsonl_file.name}")

            except Exception as e:
                print(f"Error processing {jsonl_file}: {e}")
                continue

    print(f"Successfully combined {len(jsonl_files)} files into {output_file}")
    print(f"Total lines written: {total_lines}")

    return total_lines

def main():
    """Main function to combine all HF JSONL files"""
    # Define paths using settings
    hf_dir = config.data_raw_dir / "hf"
    output_file = config.get_raw_file_path("hf_api_raw.jsonl")

    print(f"Looking for JSONL files in: {hf_dir}")
    print(f"Output will be saved to: {output_file}")

    # Find all JSONL files
    jsonl_files = find_all_jsonl_files(hf_dir)

    if not jsonl_files:
        print("No JSONL files found to combine")
        return

    # List all files to be combined
    print("\nFiles to combine:")
    for i, file in enumerate(jsonl_files, 1):
        print(f"  {i}. {file.name}")

    # Combine all files
    total_lines = combine_jsonl_files(jsonl_files, output_file)

    print(f"\n✓ Combination complete!")
    print(f"Combined {len(jsonl_files)} files")
    print(f"Total entries: {total_lines}")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()