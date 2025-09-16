import torch
import json
import sys
from pathlib import Path
from transformers import BitsAndBytesConfig, pipeline

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an expert programming instructor. For each coding problem, provide a comprehensive solution following this exact format:\n\n1) Plan\n2) Pseudocode\n3) Dry Run Trace\n4) Final Code (Python)\n5) Complexity\n6) Common Mistakes\n\nBe detailed and educational in your explanations."
}

def generate_solution(llm, problem_data, temperature=0.7, top_p=0.95, max_new_tokens=1500):
    """Generate structured solution using Qwen model"""

    # Prepare user prompt with problem and reference solution
    user_content = f"""Problem: {problem_data['question']}

Reference Solution:
```python
{problem_data['solution']}
```

Please provide a comprehensive educational solution following the 6-step format."""

    user_text = [{"role": "user", "content": user_content}]

    # Combine system prompt with user text
    prompt_context = [SYSTEM_PROMPT] + user_text

    # Apply chat template
    prompt = llm.tokenizer.apply_chat_template(
        prompt_context,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate response
    outputs = llm(prompt,
                 max_new_tokens=max_new_tokens,
                 do_sample=True,
                 temperature=temperature,
                 top_p=top_p,
                 return_full_text=False,
                 eos_token_id=[
                     llm.tokenizer.eos_token_id,
                     llm.tokenizer.convert_tokens_to_ids("<|im_end|>")],
                 pad_token_id=llm.tokenizer.eos_token_id
    )

    return outputs[0]["generated_text"].strip()

def load_seen_sft_names():
    """Load already processed SFT names to avoid duplicates"""
    seen_file = config.get_raw_file_path("taco_sft_seen.jsonl")
    seen_names = set()

    if seen_file.exists():
        with open(seen_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    seen_names.add(data.get("name", ""))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(seen_names)} previously processed SFT names")
    return seen_names

def process_taco_to_sft(llm, num_samples=None):
    """Process TACO raw data into SFT format using Qwen model"""

    # Load raw data
    raw_file = config.get_raw_file_path("taco_raw.jsonl")
    if not raw_file.exists():
        print(f"Raw data file not found: {raw_file}")
        return 0

    # Load seen names
    seen_names = load_seen_sft_names()

    # Read all raw data
    raw_data = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                raw_data.append(data)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(raw_data)} raw samples")

    # Filter out already processed samples
    if num_samples is None:
        samples_to_process = [d for d in raw_data if d.get("name", "") not in seen_names]
    else:
        samples_to_process = [d for d in raw_data if d.get("name", "") not in seen_names][:num_samples]

    print(f"Processing {len(samples_to_process)} new samples...")

    # Output files
    sft_file = config.get_raw_file_path("taco_sft.jsonl")
    seen_file = config.get_raw_file_path("taco_sft_seen.jsonl")

    processed_count = 0

    for i, sample in enumerate(samples_to_process):
        print(f"\nProcessing {i+1}/{len(samples_to_process)}: {sample.get('name', 'unnamed')}")

        try:
            # Generate structured solution using Qwen
            generated_response = generate_solution(llm, sample)

            # Create SFT format
            system_content = "You are a helpful programming assistant. Help solve coding problems step by step with detailed explanations."
            user_content = f"Problem: {sample['question']}"
            assistant_content = generated_response

            conversation_text = f"<|system|>{system_content}<|user|>{user_content}<|assistant|>{assistant_content}"

            sft_entry = {
                "text": conversation_text,
                "name": sample.get("name", f"problem_{i}")
            }

            # Append to SFT file
            with open(sft_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sft_entry, ensure_ascii=False) + '\n')

            # Track processed name
            with open(seen_file, 'a', encoding='utf-8') as f:
                seen_entry = {"name": sample.get("name", f"problem_{i}")}
                f.write(json.dumps(seen_entry, ensure_ascii=False) + '\n')

            processed_count += 1
            print(f"✓ Successfully processed: {sample.get('name', 'unnamed')}")

        except Exception as e:
            print(f"✗ Error processing {sample.get('name', 'unnamed')}: {e}")
            continue

    print(f"\n✓ SFT data generation complete!")
    print(f"Processed: {processed_count} samples")
    print(f"Output: {sft_file}")
    return processed_count

def main():
    """Main function to load model and process data"""

    # Configure quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )

    print("Loading Qwen model...")

    # Load Qwen model
    llm = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-7B-Instruct",
        model_kwargs={
            "quantization_config": quantization_config,
            "device_map": "auto",
        }
    )

    print("Model loaded successfully!")

    # Process all samples (set num_samples=10 for testing)
    count = process_taco_to_sft(llm, num_samples=None)
    print(f"SFT data generation complete: {count} samples processed")

if __name__ == "__main__":
    main()