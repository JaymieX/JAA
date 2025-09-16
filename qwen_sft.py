from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
import sys
from pathlib import Path

# Add current directory to path to import settings
sys.path.append(str(Path(__file__).parent))
from settings import config

def fine_tune_qwen_model(
    model_name: str = "unsloth/Qwen2.5-7B-Instruct",
    dataset_file: str = None,
    output_dir: str = "Qwen2.5-7B-qlora-finetuned",
    batch_size: int = 4,
    gradient_steps: int = 4,
    epochs: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048
):
    """
    Fine-tune Qwen2.5-7B-Instruct model using QLoRA with Unsloth

    Args:
        model_name: Hugging Face model name or path
        dataset_file: Path to JSONL dataset file (defaults to sft_all.jsonl)
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
        dataset_file = config.get_raw_file_path("sft_all.jsonl")

    print(f"Loading dataset: {dataset_file}")
    # Load our SFT dataset
    dataset = load_dataset("json", data_files=str(dataset_file), split="train")

    print(f"Dataset size: {len(dataset)} samples")

    # Initialize the trainer for Supervised Fine-Tuning (SFT)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_steps,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=50,
            save_strategy="epoch",
            optim="adamw_8bit",  # Use 8-bit Adam optimizer
            warmup_steps=10,
            save_total_limit=2,
        )
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

    print("=== Qwen2.5-7B-Instruct SFT Training ===")
    print(f"Dataset: {config.get_raw_file_path('sft_all.jsonl')}")
    print(f"Model: unsloth/Qwen2.5-7B-Instruct")
    print()

    # Start fine-tuning with default parameters
    model, tokenizer = fine_tune_qwen_model(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        output_dir="Qwen2.5-7B-cs-helper-finetuned",
        batch_size=2,  # Smaller batch size for 7B model
        gradient_steps=8,  # Higher gradient accumulation
        epochs=3,
        learning_rate=2e-4,
        max_seq_length=2048
    )

    print("\n=== Training Complete ===")
    print("Model saved to: Qwen2.5-7B-cs-helper-finetuned")

if __name__ == "__main__":
    main()