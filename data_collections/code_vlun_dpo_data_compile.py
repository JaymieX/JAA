import json
import random
import sys
from pathlib import Path

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config
from sft_sys_prompts import SYSTEM_PROMPTS

RNG_SEED = 1337

def convert_code_vlun_dpo_to_sft():
    """Convert code_vlun_dpo raw data to SFT format"""

    print("Converting code_vlun_dpo Test Suite data to SFT format...")

    # Input and output files
    input_file = config.get_raw_file_path("code_vlun_dpo_raw.jsonl")
    output_file = config.get_raw_file_path("code_vlun_dpo_sft.jsonl")

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return 0

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    rng = random.Random(RNG_SEED)
    processed_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for i, line in enumerate(infile):
            try:
                data = json.loads(line.strip())

                # Extract data
                user_content = data.get("rejected", "")    # vulnerable code
                assistant_content = data.get("chosen", "") # fixed code

                # Skip if missing essential data
                if not user_content or not assistant_content:
                    print(f"Skipping entry {i+1} - missing rejected or chosen")
                    continue
                
                system_content = rng.choice(SYSTEM_PROMPTS)["content"]

                # --- Write Unsloth SFT "conversations" format ---
                sft_entry = {
                    "name": f"code_vlun_dpo_{i+1}",
                    "conversations": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
                # ----------------------------------------------------------------

                # Write to output file
                outfile.write(json.dumps(sft_entry, ensure_ascii=False) + '\n')
                processed_count += 1

                # Progress indicator
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} entries...")

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON at line {i+1}: {e}")
                continue
            except Exception as e:
                print(f"Error processing entry {i+1}: {e}")
                continue

    print(f"\nO Conversion complete!")
    print(f"Processed: {processed_count} entries")
    print(f"Output saved to: {output_file}")

    return processed_count

def main():
    """Main function to convert code_vlun_dpo data to SFT format"""

    print("=== code_vlun_dpo to SFT Conversion ===")
    print("Input: code_vlun_dpo_raw.jsonl")
    print("Output: code_vlun_dpo_sft.jsonl")
    print("Format: rejected -> user, chosen -> assistant")
    print()

    # Convert the data
    count = convert_code_vlun_dpo_to_sft()

    print(f"\n=== Conversion Complete ===")
    print(f"Successfully converted {count} entries")

if __name__ == "__main__":
    main()