from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from llm_profiles import LLMProFile


def load_llm(profile : LLMProFile):
    if profile is LLMProFile.SMALL:
        print("Model Size : Small - unsloth/Qwen2.5-3B-Instruct Quantized")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        return pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct", #3B
            model_kwargs={
                "quantization_config": quantization_config,
                "device_map": "auto"
            }
        )
        
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
        
        return pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)
        
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
        
        return pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)
        
    else:
        return None