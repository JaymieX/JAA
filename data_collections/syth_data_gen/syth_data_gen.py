"""Lightweight synthetic data generator built around the Qwen inference pipeline.

`SythDataGen` keeps the model loaded once and exposes a simple API to generate
synthetic user/assistant exchanges while letting callers manage system prompts.
"""

import random
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union

import torch
from transformers import BitsAndBytesConfig, GenerationConfig, pipeline
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

    def generate(
        self,
        seed_conversation: List[Dict[str, str]],
        variations: int = 3,
        rng_seed: Optional[int] = None,
    ) -> List[List[Dict[str, str]]]:
        """Return synthetic conversations built from independently generated user/assistant turns."""
        if variations <= 0:
            return []

        user_seed = self._get_latest_content(seed_conversation, "user")
        assistant_seed = self._get_latest_content(seed_conversation, "assistant")

        rng = random.Random(rng_seed) if rng_seed is not None else random
        results: List[List[Dict[str, str]]] = []

        for _ in range(variations):
            user_variant = self._generate_user_variant(user_seed)
            assistant_variant = self._generate_assistant_variant(assistant_seed)
            system_message = self._choose_system_prompt(rng)
            results.append(
                [
                    system_message,
                    {"role": "user", "content": user_variant},
                    {"role": "assistant", "content": assistant_variant},
                ]
            )

        return results

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
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.05,
            eos_token_id=eos_ids,
        )

    def _get_latest_content(self, messages: List[Dict[str, str]], role: str) -> str:
        for message in reversed(messages):
            if message.get("role") == role and isinstance(message.get("content"), str):
                return message["content"]
            
        raise ValueError(f"Seed conversation missing a '{role}' message.")

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
