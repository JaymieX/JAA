from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import Optional, Union, Type
from pydantic import BaseModel
from llm_profiles import LLMProFile


class TransformersWrapper:
    """Wrapper to make transformers pipeline compatible with vLLM interface"""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer

    def __call__(self,
                 prompt,
                 generation_config=None,
                 return_full_text=False,
                 structured_output_schema: Optional[Union[Type[BaseModel], type]] = None,
                 **kwargs):
        """
        Generate text with transformers pipeline

        Args:
            prompt: Input prompt
            generation_config: Generation configuration
            return_full_text: Whether to return full text including prompt
            structured_output_schema: Ignored for transformers (not supported)
        """
        # Note: structured_output_schema is accepted but ignored for transformers
        # This maintains API compatibility with vLLM version

        return self.pipeline(
            prompt,
            generation_config=generation_config,
            return_full_text=return_full_text,
            **kwargs
        )


def load_llm(profile : LLMProFile):
    if profile is LLMProFile.SMALL:
        print("Model Size : Small - unsloth/Qwen2.5-3B-Instruct Quantized")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct", #3B
            model_kwargs={
                "quantization_config": quantization_config,
                "device_map": "auto"
            }
        )
        return TransformersWrapper(pipe)
        
    elif profile is LLMProFile.LARGE:
        print("Model Size : Large - unsloth/Qwen2.5-7B-Instruct")
        
        # Load tokenizer
        ft_tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct")
        
        # Base Model
        ft_base_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Qwen2.5-7B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        # Load LoRA adapter
        ft_model = PeftModel.from_pretrained(ft_base_model, "Amie69/Qwen2.5-7B-cve-coder")
        
        # Verify LoRA adapter is loaded
        if hasattr(ft_model, 'peft_config') and ft_model.peft_config:
            print(f"✓ LoRA adapter loaded successfully: {list(ft_model.peft_config.keys())}")
        else:
            print("⚠️ Warning: LoRA adapter may not be loaded correctly")
        ft_model.eval() # Inference mode

        pipe = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)
        return TransformersWrapper(pipe)
        
    elif profile is LLMProFile.SUPER_LARGE:
        print("Model Size : Super Large - unsloth/Qwen2.5-14B-Instruct")
        
        # Load tokenizer
        ft_tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-14B-Instruct")
        
        # Base Model
        ft_base_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Qwen2.5-14B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        # Load LoRA adapter
        ft_model = PeftModel.from_pretrained(ft_base_model, "Amie69/Qwen2.5-14B-cve-coder")
        
        # Verify LoRA adapter is loaded
        if hasattr(ft_model, 'peft_config') and ft_model.peft_config:
            print(f"✓ LoRA adapter loaded successfully: {list(ft_model.peft_config.keys())}")
        else:
            print("⚠️ Warning: LoRA adapter may not be loaded correctly")
        ft_model.eval() # Inference mode

        pipe = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)
        return TransformersWrapper(pipe)
        
    else:
        return None