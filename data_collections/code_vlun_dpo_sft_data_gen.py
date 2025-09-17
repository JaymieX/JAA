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

SYSTEM_PROMPT = (
    "You are a senior application security engineer.\n"
    "Output ONLY valid JSON matching this schema, with no code fences and no extra text:\n"
    "{\n"
    '  "vulnerability_type": string,\n'
    '  "why_bad": string,\n'
    '  "exploit_scenario": string,\n'
    '  "evidence": [ {"line": int, "snippet": string, "reason": string} ],\n'
    '  "fix": {"strategy": string, "patched_code": string}\n'
    "}\n"
    "Rules:\n"
    "- Analyze the vulnerable_code; treat candidate_fix_code only as a suggestion.\n"
    "- If the candidate fix is unsafe/incomplete, replace it with a safer patch.\n"
    "- Cite concrete lines/patterns in evidence; if unknown, use [] or empty strings.\n"
    "- Keep exploit_scenario and why_bad short and concise\n"
    "- Escape newlines in patched_code with \\n. No markdown or fences."
)

USER_TMPL = (
    "Language: {lang}\n"
    "Vulnerability hint: {vuln}\n"
    "Vulnerable code:\n{vuln_code}\n"
    "Candidate fix code:\n{fix_code}\n"
)

def create_pipeline_input(example, tokenizer):
    """Create pipeline input string directly from raw data"""
    user_content = USER_TMPL.format(
        lang=example.get('lang', 'unknown'),
        vuln=example.get('vulnerability', ''),
        vuln_code=example.get('rejected', ''),  # vulnerable code
        fix_code=example.get('chosen', '')      # fixed code
    )

    # Create prompt context for chat template
    prompt_context = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # Apply chat template to get the final prompt string
    prompt = tokenizer.apply_chat_template(
        prompt_context,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt

def load_seen_sft_names():
    """Load already processed SFT names to avoid duplicates"""
    seen_file = config.get_raw_file_path("code_vlun_sft_seen.jsonl")
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

def process_vulnerability_to_sft_batch(llm, num_samples=None, batch_size=4):
    """Process Code Vulnerability DPO data into SFT format using Qwen model with batch processing"""

    # Load raw data
    raw_file = config.get_raw_file_path("code_vlun_dpo_raw.jsonl")
    if not raw_file.exists():
        print(f"Raw data file not found: {raw_file}")
        return 0

    # Load seen names
    seen_names = load_seen_sft_names()

    # Read all raw data
    raw_data = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                # Add unique name for tracking
                data['name'] = f"code_vlun_{i+1}"
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

    # Prepare pipeline inputs directly without storing in dataset
    print("Preparing pipeline inputs...")
    pipeline_inputs = []
    for sample in samples_to_process:
        prompt = create_pipeline_input(sample, llm.tokenizer)
        pipeline_inputs.append(prompt)

    # Output files
    sft_file = config.get_raw_file_path("code_vlun_sft.jsonl")
    seen_file = config.get_raw_file_path("code_vlun_sft_seen.jsonl")

    print(f"Starting batch processing with batch_size={batch_size}...")

    # Process with batch pipeline
    processed_count = 0
    results = []

    # Process with pipeline using direct inputs
    for i, (sample, output) in enumerate(tqdm(
        zip(samples_to_process,
            llm(pipeline_inputs,
                batch_size=batch_size,
                max_new_tokens=1000,  # Shorter for JSON format
                do_sample=False,
                temperature=0.3,  # Lower for more consistent JSON
                top_p=0.9,
                return_full_text=False,
                eos_token_id=[
                    llm.tokenizer.eos_token_id,
                    llm.tokenizer.convert_tokens_to_ids("<|im_end|>")
                ],
                pad_token_id=llm.tokenizer.eos_token_id
            )),
        total=len(samples_to_process),
        desc="Generating vulnerability analysis"
    )):
        try:
            generated_response = output[0]["generated_text"].strip()

            # Create SFT format
            system_content = SYSTEM_PROMPT

            user_content = USER_TMPL.format(
                lang=sample.get('lang', 'unknown'),
                vuln=sample.get('vulnerability', ''),
                vuln_code=sample.get('rejected', ''),  # vulnerable code
                fix_code=sample.get('chosen', '')      # fixed code
            )

            assistant_content = generated_response

            conversation_text = f"<|system|>{system_content}<|user|>{user_content}<|assistant|>{assistant_content}<|endoftext|>"

            sft_entry = {
                "text": conversation_text,
                "name": sample.get("name", f"code_vlun_{i}")
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

    # Load Qwen model optimized for V100 48GB performance
    llm = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-7B-Instruct",
        model_kwargs={
            "torch_dtype": torch.float16,  # Use FP16 for speed while maintaining quality
            "device_map": "auto",
        },
        # Pipeline optimizations for V100
        batch_size=12,  # Default batch size for pipeline
        max_length=None,  # No hard limit, let model decide
    )

    print("Model loaded successfully!")

    # Process all samples optimized for V100 48GB
    # Larger batch size for better GPU utilization
    count = process_vulnerability_to_sft_batch(llm, num_samples=None, batch_size=12)
    print(f"SFT data generation complete: {count} samples processed")

if __name__ == "__main__":
    main()