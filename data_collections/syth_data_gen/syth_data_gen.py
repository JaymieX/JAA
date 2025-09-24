"""Lightweight synthetic data generator built around the Qwen inference pipeline.

`SythDataGen` keeps the model loaded once and exposes a simple API to generate
synthetic user/assistant exchanges while letting callers manage system prompts.
"""

import random
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union

import torch
from transformers import BitsAndBytesConfig, GenerationConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import syth_gen_prompt as gen_prompts

SMALL_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LARGE_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"

DEFAULT_SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}


class ModelSize(str, Enum):
    SMALL = "small"
    LARGE = "large"


class SythDataGen:
    def __init__(self, size: Union[ModelSize, str] = ModelSize.SMALL) -> None:
        self.size = self._coerce_size(size)
        self.llm = self._load_llm(self.size)
        self.tokenizer = self.llm.tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set left padding for decoder-only models (required for batch generation)
        self.tokenizer.padding_side = 'left'

        self.gen_cfg = self._build_generation_config(self.tokenizer)
        self._system_prompts: List[Dict[str, str]] = [DEFAULT_SYSTEM_MESSAGE]

    def _coerce_size(self, size: Union[ModelSize, str]) -> ModelSize:
        if isinstance(size, ModelSize):
            return size
        
        if isinstance(size, str):
            try:
                return ModelSize(size.lower())
            except ValueError as exc:
                raise ValueError("Model size must be 'small' or 'large'.") from exc
            
        raise TypeError("Model size must be a ModelSize or string value.")

    def set_system_prompts(self, prompts: Sequence[Dict[str, str]]) -> None:
        """Store allowed system prompts that will be prepended to synthetic outputs."""
        cleaned: List[Dict[str, str]] = []
        for prompt in prompts:
            if not isinstance(prompt, dict):
                raise ValueError("System prompts must be dicts with role/content fields.")
            
            role = prompt.get("role")
            content = prompt.get("content")
            
            if role != "system" or not isinstance(content, str):
                raise ValueError("Each system prompt requires role='system' and string content.")
            
            cleaned.append({"role": "system", "content": content})
            
        if not cleaned:
            raise ValueError("At least one valid system prompt is required.")
        
        self._system_prompts = cleaned

    def generate_batch(
        self,
        seed_conversations: List[List[Dict[str, str]]],
        variations: int = 3,
        rng_seed: Optional[int] = None,
    ) -> List[List[List[Dict[str, str]]]]:
        """Process multiple seed conversations at once using Dataset API for maximum efficiency."""
        if variations <= 0 or not seed_conversations:
            return []

        # Extract all user and assistant seeds
        all_user_seeds = []
        all_assistant_seeds = []

        for seed_conversation in seed_conversations:
            user_seed = self._get_latest_content(seed_conversation, "user")
            assistant_seed = self._get_latest_content(seed_conversation, "assistant")

            # Repeat each seed for the number of variations
            all_user_seeds.extend([user_seed] * variations)
            all_assistant_seeds.extend([assistant_seed] * variations)

        # Generate ALL variations at once using Dataset
        user_variants = self._generate_all_variants_with_dataset(all_user_seeds, "user")
        assistant_variants = self._generate_all_variants_with_dataset(all_assistant_seeds, "assistant")

        # Group results back by seed
        rng = random.Random(rng_seed) if rng_seed is not None else random
        results = []

        for seed_idx in range(len(seed_conversations)):
            seed_results = []
            for var_idx in range(variations):
                global_idx = seed_idx * variations + var_idx
                system_message = self._choose_system_prompt(rng)
                seed_results.append([
                    system_message,
                    {"role": "user", "content": user_variants[global_idx]},
                    {"role": "assistant", "content": assistant_variants[global_idx]},
                ])
            results.append(seed_results)

        return results

    def generate(
        self,
        seed_conversation: List[Dict[str, str]],
        variations: int = 3,
        rng_seed: Optional[int] = None,
    ) -> List[List[Dict[str, str]]]:
        """Legacy single-seed method - calls batch method for consistency."""
        batch_results = self.generate_batch([seed_conversation], variations, rng_seed)
        return batch_results[0] if batch_results else []

    def _choose_system_prompt(self, rng) -> Dict[str, str]:
        selected = rng.choice(self._system_prompts)
        return {"role": "system", "content": selected["content"]}

    def _load_llm(self, size: ModelSize):
        """Instantiate the HF pipeline once so callers reuse the same model."""
        if size is ModelSize.SMALL:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            
            return pipeline(
                "text-generation",
                model=SMALL_MODEL_ID,
                model_kwargs={
                    "device_map": "auto",
                    "quantization_config": quantization_config,
                },
            )
            
        if size is ModelSize.LARGE:
            return pipeline(
                "text-generation",
                model=LARGE_MODEL_ID,
                model_kwargs={"device_map": "auto"},
            )
            
        raise ValueError(f"Unsupported model size: {size}")

    def _build_generation_config(self, tokenizer) -> GenerationConfig:
        """Reuse project generation settings, including EOS handling."""
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [tokenizer.eos_token_id]
        if im_end is not None:
            eos_ids.append(im_end)

        return GenerationConfig(
            max_new_tokens=400,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=eos_ids,
        )

    def _get_latest_content(self, messages: List[Dict[str, str]], role: str) -> str:
        for message in reversed(messages):
            if message.get("role") == role and isinstance(message.get("content"), str):
                return message["content"]
            
        raise ValueError(f"Seed conversation missing a '{role}' message.")

    def _generate_all_variants_with_dataset(self, source_texts: List[str], variant_type: str) -> List[str]:
        """Generate ALL variations at once using Dataset API to avoid sequential pipeline warning."""
        if not source_texts:
            return []

        # Choose appropriate system prompt based on variant type
        if variant_type == "user":
            system_prompt = gen_prompts.USER_VARIATION_SYSTEM_PROMPT
        else:
            system_prompt = gen_prompts.ASSISTANT_VARIATION_SYSTEM_PROMPT

        # Create prompts for ALL source texts
        prompts = []
        for i, source_text in enumerate(source_texts):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": source_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

            # Debug: Print first prompt to verify system prompt is applied correctly
            # if i == 0:
            #     print(f"FULL PROMPT SENT TO MODEL ({variant_type}):\n{prompt}\n{'='*50}")

        # Convert to Dataset
        dataset = Dataset.from_dict({"text": prompts})

        # Use pipeline with KeyDataset for efficient batching
        results = []
        batch_size = min(len(prompts), 8)  # Reasonable batch size

        print(f"Processing {len(prompts)} {variant_type} variations in batches of {batch_size}")

        for output in self.llm(KeyDataset(dataset, "text"),
                              generation_config=self.gen_cfg,
                              return_full_text=False,
                              batch_size=batch_size):
            # KeyDataset always returns list format: [{"generated_text": "..."}]
            if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                cleaned_text = self._clean_generated_text(output[0]["generated_text"].strip())
            else:
                # Fallback for unexpected formats
                print(f"WARNING: Unexpected output format: {type(output)}")
                cleaned_text = str(output).strip()

            results.append(cleaned_text)

        return results

    def _generate_user_variant(self, source_text: str) -> str:
        prompt_messages = [
            {"role": "system", "content": gen_prompts.USER_VARIATION_SYSTEM_PROMPT},
            {"role": "user", "content": source_text},
        ]
        
        raw = self._run_chat(prompt_messages)
        return self._clean_generated_text(raw)

    def _generate_assistant_variant(self, source_text: str) -> str:
        prompt_messages = [
            {"role": "system", "content": gen_prompts.ASSISTANT_VARIATION_SYSTEM_PROMPT},
            {"role": "user", "content": source_text},
        ]
        
        raw = self._run_chat(prompt_messages)
        return self._clean_generated_text(raw)

    def _run_chat(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        output = self.llm(prompt, generation_config=self.gen_cfg, return_full_text=False)
        return output[0]["generated_text"].strip()

    def _clean_generated_text(self, text: str) -> str:
        cleaned = text.strip()
        
        if cleaned.startswith("```"):
            parts = cleaned.split("```", 2)
            cleaned = parts[1].strip() if len(parts) >= 2 else cleaned
            
        return cleaned
