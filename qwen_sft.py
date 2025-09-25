from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import sys
from pathlib import Path

# NEW: import chat template helper
from unsloth.chat_templates import get_chat_template

# Add current directory to path to import settings
sys.path.append(str(Path(__file__).parent))
from settings import config

def fine_tune_qwen_model(
    model_name: str = "unsloth/Qwen2.5-14B-Instruct",
    dataset_file: str = None,
    output_dir: str = "Qwen2.5-14B-qlora-finetuned",
    batch_size: int = 4,
    gradient_steps: int = 4,
    epochs: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048
):
    """
    Fine-tune Qwen2.5-14B-Instruct model using QLoRA with Unsloth

    Args:
        model_name: Hugging Face model name or path
        dataset_file: Path to JSONL dataset file (defaults to sft_cve.jsonl)
        output_dir: Directory to save fine-tuned model
        batch_size: Per-device training batch size
        gradient_steps: Gradient accumulation steps
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        max_seq_length: Maximum sequence length for training

    Returns:
        Tuple of (model, tokenizer) after training
    """

    print(f"Loading model: {model_name}")
    # Load the base Qwen model in 4-bit mode (dynamic quantization)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True
    )

    # QLORA Adaptor
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )

    # Set default dataset file if not provided
    if dataset_file is None:
        dataset_file = config.get_raw_file_path("sft_cve.jsonl")

    print(f"Loading dataset: {dataset_file}")
    # Load our SFT dataset with shuffling
    dataset = load_dataset("json", data_files=str(dataset_file), split="train")

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=42)

    print(f"Dataset size: {len(dataset)} samples (shuffled)")

    # --- NEW: turn conversations -> text via Qwen-2.5 chat template ---
    chat_tmpl = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def to_text(example):
        conv = example.get("conversations")
        if not conv:
            # Fallback if a row is in old {text: ...} format
            return {"text": example.get("text", "")}
        
        # Produce the fully formatted prompt+answer text
        txt = chat_tmpl.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,  # we already include assistant messages
        )
        return {"text": txt}


    dataset = dataset.map(to_text)
    
    # ---------------------------------------------------------------
    
    train_cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=25,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        optim="adamw_8bit",
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        packing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        group_by_length=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=train_cfg,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Fine-tuning completed!")
    return model, tokenizer

def main():
    """Main function to start SFT training"""

    print("=== Qwen2.5-14B-Instruct SFT Training ===")
    print(f"Dataset: {config.get_raw_file_path('sft_cve.jsonl')}")
    print(f"Model: unsloth/Qwen2.5-14B-Instruct")
    print()

    # Start fine-tuning optimized for V100 48GB
    model, tokenizer = fine_tune_qwen_model(
        model_name="unsloth/Qwen2.5-14B-Instruct",
        dataset_file=str(config.get_raw_file_path("sft_cve.jsonl")),
        output_dir="Qwen2.5-14B-cs-coder-finetuned",
        batch_size=1,  # Larger batch size for V100 48GB
        gradient_steps=32,  # Effective batch size 32
        epochs=2,  # Fewer epochs for large dataset
        learning_rate=5e-5,  # Lower LR for large dataset stability
        max_seq_length=8192  # Longer sequences for complex code problems
    )

    print("\n=== Training Complete ===")
    print("Model saved to: Qwen2.5-14B-cs-coder-finetuned")

if __name__ == "__main__":
    main()