import torch
import json
import sys
from pathlib import Path
from transformers import BitsAndBytesConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from tqdm.auto import tqdm

# Add parent directory to path to import settings
sys.path.append(str(Path(__file__).parent.parent))
from settings import config

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an expert programming instructor. For each coding problem, provide a comprehensive solution following this exact format:\n\n1) Plan\n2) Pseudocode\n3) Dry Run Trace\n4) Final Code (Python)\n5) Complexity\n6) Common Mistakes\n\nBe detailed and educational in your explanations."
}

def format_prompt(example):
    """Format problem data into prompt for the model"""
    user_content = f"""Problem: {example['question']}

Reference Solution:
```python
{example['solution']}
```

Please provide a comprehensive educational solution following the 6-step format."""

    # Combine system prompt with user text
    prompt_context = [SYSTEM_PROMPT, {"role": "user", "content": user_content}]

    # Store formatted prompt in the example
    example['formatted_prompt'] = prompt_context
    return example

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

def process_taco_to_sft_batch(llm, num_samples=None, batch_size=8):
    """Process TACO raw data into SFT format using Qwen model with batch processing"""

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

    if not samples_to_process:
        print("No new samples to process!")
        return 0

    # Convert to HuggingFace Dataset for efficient batch processing
    dataset = Dataset.from_list(samples_to_process)

    # Format prompts for all samples
    print("Formatting prompts...")
    dataset = dataset.map(format_prompt, batched=False)

    # Create custom dataset field for pipeline input
    def create_pipeline_input(example):
        # Apply chat template to get the final prompt string
        prompt = llm.tokenizer.apply_chat_template(
            example['formatted_prompt'],
            tokenize=False,
            add_generation_prompt=True
        )
        example['pipeline_input'] = prompt
        return example

    dataset = dataset.map(create_pipeline_input, batched=False)

    # Output files
    sft_file = config.get_raw_file_path("taco_sft.jsonl")
    seen_file = config.get_raw_file_path("taco_sft_seen.jsonl")

    print(f"Starting batch processing with batch_size={batch_size}...")

    # Process with batch pipeline
    processed_count = 0
    results = []

    # Use KeyDataset for efficient pipeline processing
    for i, (sample, output) in enumerate(tqdm(
        zip(samples_to_process,
            llm(KeyDataset(dataset, "pipeline_input"),
                batch_size=batch_size,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                return_full_text=False,
                eos_token_id=[
                    llm.tokenizer.eos_token_id,
                    llm.tokenizer.convert_tokens_to_ids("<|im_end|>")
                ],
                pad_token_id=llm.tokenizer.eos_token_id
            )),
        total=len(samples_to_process),
        desc="Generating solutions"
    )):
        try:
            generated_response = output[0]["generated_text"].strip()

            # Create SFT format
            system_content = "You are a helpful programming assistant. Help solve coding problems step by step with detailed explanations."
            user_content = f"Problem: {sample['question']}"
            assistant_content = generated_response

            conversation_text = f"<|system|>{system_content}<|user|>{user_content}<|assistant|>{assistant_content}"

            sft_entry = {
                "text": conversation_text,
                "name": sample.get("name", f"problem_{i}")
            }

            results.append(sft_entry)
            processed_count += 1

        except Exception as e:
            print(f"✗ Error processing {sample.get('name', 'unnamed')}: {e}")
            continue

    # Write all results at once for better performance
    print(f"\nWriting {len(results)} results to files...")

    with open(sft_file, 'a', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(seen_file, 'a', encoding='utf-8') as f:
        for entry in results:
            seen_entry = {"name": entry["name"]}
            f.write(json.dumps(seen_entry, ensure_ascii=False) + '\n')

    print(f"\n✓ SFT data generation complete!")
    print(f"Processed: {processed_count} samples")
    print(f"Output: {sft_file}")
    return processed_count

def main():
    """Main function to load model and process data"""

    print("Loading Qwen model at full precision...")

    # Load Qwen model without quantization for full performance
    llm = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-7B-Instruct",
        model_kwargs={
            "torch_dtype": torch.float16,  # Use FP16 for speed while maintaining quality
            "device_map": "auto",
        }
    )

    print("Model loaded successfully!")

    # Process all samples using optimized batch processing
    # Start with smaller batch_size (4-8) and increase if GPU memory allows
    count = process_taco_to_sft_batch(llm, num_samples=None, batch_size=4)
    print(f"SFT data generation complete: {count} samples processed")

if __name__ == "__main__":
    main()