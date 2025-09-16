import json
import sys
from pathlib import Path

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

def load_raw_data(raw_file_path):
    """Load raw data from a JSONL file"""
    data = []

    if not raw_file_path.exists():
        print(f"Raw file not found: {raw_file_path}")
        return data

    print(f"Loading data from: {raw_file_path}")

    try:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                        line_count += 1
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON line: {e}")
                        continue

        print(f"Loaded {line_count} entries from {raw_file_path.name}")

    except Exception as e:
        print(f"Error loading {raw_file_path}: {e}")

    return data

def convert_to_sft_format(raw_data, system_content):
    """Convert raw data to SFT conversation format"""
    sft_data = []

    for entry in raw_data:
        # Extract fields
        name = entry.get("name", "unknown")
        question = entry.get("question", "")
        solution = entry.get("solution", "")

        # Skip entries with missing data
        if not question or not solution:
            print(f"Skipping entry {name} - missing question or solution")
            continue

        # Create SFT conversation format
        user_content = question
        assistant_content = solution

        conversation_text = f"<|system|>{system_content}<|user|>{user_content}<|assistant|>{assistant_content}<|endoftext|>"

        sft_entry = {
            "text": conversation_text,
            "name": name
        }

        sft_data.append(sft_entry)

    return sft_data

def process_raw_files(raw_items, system_content, output_file):
    """Process multiple raw files and combine into SFT format"""
    all_raw_data = []

    print(f"Processing {len(raw_items)} raw files...")

    # Load data from all specified files
    for raw_item in raw_items:
        raw_file_path = config.get_raw_file_path(raw_item)
        raw_data = load_raw_data(raw_file_path)
        all_raw_data.extend(raw_data)

    print(f"Total entries loaded: {len(all_raw_data)}")

    # Convert to SFT format
    print("Converting to SFT format...")
    sft_data = convert_to_sft_format(all_raw_data, system_content)

    # Save to output file
    print(f"Saving {len(sft_data)} SFT entries to: {output_file}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in sft_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"✓ Successfully saved {len(sft_data)} entries to {output_file}")
        return len(sft_data)

    except Exception as e:
        print(f"Error saving to {output_file}: {e}")
        return 0

def main():
    """Main function to convert raw data to SFT format"""

    # System prompt (customize this as needed)
    system_content = """You are a helpful Python assistant.
Write a single, correct, efficient Python 3 solution for the given task.

Rules:
- Output ONLY Python code (no prose, no markdown fences).
- Follow the task's I/O spec exactly:
  • If a function signature is given, implement it.
  • Otherwise read from stdin and write to stdout.
- Use the standard library unless the prompt explicitly allows others.
- Be deterministic; no debug prints or extra logging.
- Handle typical edge cases and large inputs.
"""

    # List of raw JSONL files to include (manually enter here)
    raw_items = [
        "taco_raw.jsonl",
        "hf_api_raw.jsonl",
        # Add more raw files here as needed
    ]

    # Output file
    output_file = config.get_raw_file_path("sft_all.jsonl")

    print("=== Raw to SFT Conversion ===")
    print(f"System prompt: {system_content}")
    print(f"Files to process: {raw_items}")
    print(f"Output file: {output_file}")
    print()

    # Process all files
    total_entries = process_raw_files(raw_items, system_content, output_file)

    print(f"\n=== Conversion Complete ===")
    print(f"Total entries processed: {total_entries}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()