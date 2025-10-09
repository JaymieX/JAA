import json
import random
from pathlib import Path
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from eval_llm_loader import load_llm

# Fixed seed for reproducibility
RANDOM_SEED = 42


def load_eval_dataset(dataset_path):
    """Load the SFT evaluation dataset"""
    print(f"Loading evaluation dataset from {dataset_path}...")

    conversations = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            conversations.append(data['conversations'])

    print(f"Loaded {len(conversations)} conversations")
    return conversations


def format_prompt_for_model(system_msg, user_msg, tokenizer):
    """Format the prompt using the chat template"""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def calculate_metrics(reference, candidate, rouge_scorer_obj, smoothing):
    """Calculate ROUGE and BLEU scores"""

    # ROUGE scores
    rouge_scores = rouge_scorer_obj.score(reference, candidate)

    # BLEU score (tokenize by whitespace)
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    # Use smoothing to handle edge cases
    bleu_score = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        smoothing_function=smoothing.method4
    )

    return {
        'rouge1_f1': rouge_scores['rouge1'].fmeasure,
        'rouge2_f1': rouge_scores['rouge2'].fmeasure,
        'rougeL_f1': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu_score
    }


def evaluate_model(model_wrapper, conversations, tokenizer, use_lora=False, max_samples=None, seed=RANDOM_SEED):
    """Evaluate model on the dataset"""

    model_type = "SFT" if use_lora else "Base"
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type} Model")
    print(f"{'='*60}")

    # Initialize metrics
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction()

    all_metrics = {
        'rouge1_f1': [],
        'rouge2_f1': [],
        'rougeL_f1': [],
        'bleu': []
    }

    # Sample randomly with fixed seed if max_samples specified
    if max_samples and max_samples < len(conversations):
        random.seed(seed)
        eval_conversations = random.sample(conversations, max_samples)
        print(f"Randomly sampled {max_samples} conversations (seed={seed})")
    else:
        eval_conversations = conversations

    for idx, conversation in enumerate(eval_conversations):
        # Parse conversation
        system_msg = conversation[0]['content']
        user_msg = conversation[1]['content']
        reference_response = conversation[2]['content']

        # Format prompt
        prompt = format_prompt_for_model(system_msg, user_msg, tokenizer)

        # Generate response
        output = model_wrapper(
            prompt,
            generation_config=None,
            return_full_text=False,
            use_lora=use_lora
        )

        generated_response = output[0]['generated_text'].strip()

        # Calculate metrics
        metrics = calculate_metrics(
            reference_response,
            generated_response,
            rouge_scorer_obj,
            smoothing
        )

        # Store metrics
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        # Progress update
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(eval_conversations)} samples...")

    # Calculate averages
    avg_metrics = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in all_metrics.items()
    }

    print(f"\n{model_type} Model Results:")
    print(f"  ROUGE-1 F1: {avg_metrics['rouge1_f1']:.4f}")
    print(f"  ROUGE-2 F1: {avg_metrics['rouge2_f1']:.4f}")
    print(f"  ROUGE-L F1: {avg_metrics['rougeL_f1']:.4f}")
    print(f"  BLEU Score: {avg_metrics['bleu']:.4f}")

    return {
        'model_type': model_type,
        'avg_metrics': avg_metrics,
        'all_metrics': all_metrics,
        'num_samples': len(eval_conversations)
    }


def main():
    # Get paths relative to this script's location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # llm/eval -> llm -> project_root

    dataset_path = project_root / "data" / "raw" / "sft_cve.jsonl"
    results_path = script_dir / "eval_results.json"

    # Load model
    print("Loading model...")
    model_wrapper = load_llm()

    # Load dataset
    conversations = load_eval_dataset(dataset_path)

    # Evaluate on a subset for demo (adjust max_samples as needed)
    # Set to None to evaluate on entire dataset
    max_samples = 100  # Start with 100 samples for quick demo

    print(f"\nEvaluating on {max_samples if max_samples else 'all'} samples...")

    # Evaluate base model (without LoRA)
    base_results = evaluate_model(
        model_wrapper,
        conversations,
        model_wrapper.tokenizer,
        use_lora=False,
        max_samples=max_samples
    )

    # Evaluate SFT model (with LoRA)
    sft_results = evaluate_model(
        model_wrapper,
        conversations,
        model_wrapper.tokenizer,
        use_lora=True,
        max_samples=max_samples
    )

    # Compare results
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Base':<12} {'SFT':<12} {'Improvement':<12}")
    print(f"{'-'*60}")

    for metric in ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bleu']:
        base_score = base_results['avg_metrics'][metric]
        sft_score = sft_results['avg_metrics'][metric]
        improvement = ((sft_score - base_score) / base_score * 100) if base_score > 0 else 0

        print(f"{metric:<15} {base_score:<12.4f} {sft_score:<12.4f} {improvement:+.2f}%")

    # Save results
    results = {
        'base_model': base_results,
        'sft_model': sft_results,
        'dataset_path': str(dataset_path),
        'num_samples_evaluated': max_samples,
        'random_seed': RANDOM_SEED
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
