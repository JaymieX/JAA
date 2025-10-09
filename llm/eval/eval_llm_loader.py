from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


MAX_VRAM_USAGE_PERCENTAGE = 0.8


class VLLMWrapper:
    """Wrapper to make vLLM compatible with transformers pipeline interface"""
    def __init__(self, llm, tokenizer, lora_request=None):
        self.llm          = llm
        self.tokenizer    = tokenizer
        self.lora_request = lora_request

    def __call__(self,
                 prompt,
                 generation_config=None,
                 return_full_text=False,
                 use_lora=False):
        """
        Generate text with optional structured output

        Args:
            prompt: Input prompt
            generation_config: Generation configuration
            return_full_text: Whether to return full text including prompt
        """

        # Convert transformers GenerationConfig to vLLM SamplingParams
        sampling_params = SamplingParams(
            max_tokens         = generation_config.max_new_tokens if generation_config else 512,
            temperature        = 0.0 if (generation_config and not generation_config.do_sample) else 0.7,
            repetition_penalty = generation_config.repetition_penalty if generation_config else 1.0,
            stop_token_ids     = generation_config.eos_token_id if generation_config else None
        )
        
        lora_request = self.lora_request if use_lora else None

        # Generate with vLLM
        outputs = self.llm.generate(
            [prompt],
            sampling_params = sampling_params,
            lora_request    = lora_request
        )

        # Format output to match transformers pipeline
        # vLLM returns only generated text by default (like return_full_text=False)
        generated_text = outputs[0].outputs[0].text

        # If return_full_text=True, prepend the prompt
        if return_full_text:
            generated_text = prompt + generated_text

        return [{"generated_text": generated_text}]


def load_llm():
    print("Model Size : Super Large - unsloth/Qwen2.5-14B-Instruct + LoRA (vLLM)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-14B-Instruct")
    
    # Load base model with vLLM and enable LoRA
    llm = LLM(
        model="unsloth/Qwen2.5-14B-Instruct",
        dtype="half",
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=MAX_VRAM_USAGE_PERCENTAGE
    )
    
    # Create LoRA request
    lora_request = LoRARequest(
        lora_name="cve-coder",
        lora_int_id=1,
        lora_path="Amie69/Qwen2.5-14B-cve-coder"
    )
    
    # Verify LoRA adapter by checking if it can be used
    try:
        test_output = llm.generate(
            ["Test"],
            sampling_params=SamplingParams(max_tokens=1, temperature=0),
            lora_request=lora_request
        )
        print(f"✓ LoRA adapter loaded successfully: cve-coder")
        
    except Exception as e:
        print(f"⚠️ Warning: LoRA adapter may not be loaded correctly: {e}")
        
    return VLLMWrapper(llm, tokenizer, lora_request)