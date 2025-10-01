from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
from enum import Enum

class LLMProFile(Enum):
    SMALL       = 0,
    LARGE       = 1,
    SUPER_LARGE = 2


class VLLMWrapper:
    """Wrapper to make vLLM compatible with transformers pipeline interface"""
    def __init__(self, llm, tokenizer, lora_request=None):
        self.llm          = llm
        self.tokenizer    = tokenizer
        self.lora_request = lora_request

    def __call__(self, prompt, generation_config=None, return_full_text=False, **kwargs):
        # Convert transformers GenerationConfig to vLLM SamplingParams
        sampling_params = SamplingParams(
            max_tokens         = generation_config.max_new_tokens if generation_config else 512,
            temperature        = 0.0 if (generation_config and not generation_config.do_sample) else 0.7,
            repetition_penalty = generation_config.repetition_penalty if generation_config else 1.0,
            stop_token_ids     = generation_config.eos_token_id if generation_config else None
        )

        # Generate with vLLM
        outputs = self.llm.generate(
            [prompt],
            sampling_params = sampling_params,
            lora_request    = self.lora_request
        )

        # Format output to match transformers pipeline
        # vLLM returns only generated text by default (like return_full_text=False)
        generated_text = outputs[0].outputs[0].text

        # If return_full_text=True, prepend the prompt
        if return_full_text:
            generated_text = prompt + generated_text

        return [{"generated_text": generated_text}]


def load_llm(profile : LLMProFile):
    if profile is LLMProFile.SMALL:
        print("Model Size : Small - Qwen/Qwen2.5-3B-Instruct (vLLM)")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

        # Load model with vLLM
        llm = LLM(
            model="Qwen/Qwen2.5-3B-Instruct",
            dtype="half",
            gpu_memory_utilization=0.9
        )

        return VLLMWrapper(llm, tokenizer)
        
    elif profile is LLMProFile.LARGE:
        print("Model Size : Large - unsloth/Qwen2.5-7B-Instruct + LoRA (vLLM)")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct")

        # Load base model with vLLM and enable LoRA
        llm = LLM(
            model="unsloth/Qwen2.5-7B-Instruct",
            dtype="half",
            enable_lora=True,
            max_lora_rank=64,
            gpu_memory_utilization=0.9
        )

        # Create LoRA request
        lora_request = LoRARequest(
            lora_name="cve-coder",
            lora_int_id=1,
            lora_path="Amie69/Qwen2.5-7B-cve-coder"
        )

        print(f"✓ LoRA adapter configured: cve-coder")

        return VLLMWrapper(llm, tokenizer, lora_request)
        
    elif profile is LLMProFile.SUPER_LARGE:
        print("Model Size : Super Large - unsloth/Qwen2.5-14B-Instruct + LoRA (vLLM)")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-14B-Instruct")

        # Load base model with vLLM and enable LoRA
        llm = LLM(
            model="unsloth/Qwen2.5-14B-Instruct",
            dtype="half",
            enable_lora=True,
            max_lora_rank=64,
            gpu_memory_utilization=0.9
        )

        # Create LoRA request
        lora_request = LoRARequest(
            lora_name="cve-coder",
            lora_int_id=1,
            lora_path="Amie69/Qwen2.5-14B-cve-coder"
        )

        print(f"✓ LoRA adapter configured: cve-coder")

        return VLLMWrapper(llm, tokenizer, lora_request)

    else:
        return None