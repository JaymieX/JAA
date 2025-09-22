import json
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig, pipeline
from enum import Enum

from summarize import Summarizer
from search import Search
from notion import Notion

class LLMProFile(Enum):
    SMALL = 0,
    LARGE = 1
    

SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a helpful AI assistant with access to tools. The user will provide you with text that has been transcribed from an audio file. Your job is to have a friendly but shy response to the content of the transcription.

    You have access to the following tools:
    1. search_arxiv(query) - for searching arXiv papers
    2. notion() - for sending content to Notion

    When the user asks to search arXiv papers or research, respond with a JSON function call in this exact format:
    {"function": "search_arxiv", "arguments": {"query": "the search query"}}

    When the user asks to create a Notion note, respond with a JSON function call in this exact format:
    {"function": "notion"}

    For all other conversations, respond normally with friendly text (no JSON).

    Examples:
    User: "Search for papers about quantum computing"
    Assistant: {"function": "search_arxiv", "arguments": {"query": "quantum computing"}}

    User: "Save this to Notion"
    Assistant: {"function": "notion"}

    User: "How are you today?"
    Assistant: Oh, hello! I'm doing well, thank you for asking. How are you?"""
}

class LLM:
    def __init__(self, profile : LLMProFile, notion_token, notion_page_id):
        self.conversation_history = []
        
        if profile == LLMProFile.SMALL:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            
            self.llm = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-3B-Instruct", #3B
                model_kwargs={
                    "quantization_config": quantization_config,
                    "device_map": "auto"
                }
            )
            
        elif profile == LLMProFile.LARGE:
            self.llm = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-7B-Instruct", #7B
                model_kwargs={
                    "device_map": "auto"
                }
            )
            
        else:
            print("LLM fail to load.")
            return
            
        
        print("LLM loaded.")
        
        self.rag_search = Search()
        self.summarizer = Summarizer()
        self.notion     = Notion(notion_token, notion_page_id)
        
        self.is_notion_connected = self.notion.is_connected()
        
        
    def route_llm_output(self, llm_output: str) -> str:
        """
        Route LLM response to the correct tool if it's a function call, else return the text.
        Expects LLM output in JSON format like {'function': ..., 'arguments': {...}}.
        """
        try:
            output = json.loads(llm_output)
            func_name = output.get("function")
            args = output.get("arguments", {})
            
        except (json.JSONDecodeError, TypeError):
            # Not a JSON function call; return the text directly
            return llm_output

        if func_name == "search_arxiv":
            query = args.get("query", "")
            print(f"FUNCTION CALL: search_arxiv(query='{query}')")
            
            # Get top x rag paper and combine
            rag_results = self.rag_search.hybrid_search(query, 3)
            combined_text = ""
            for search_hit in rag_results["hits"]:
                print(f"HIT TEXT: {search_hit.text}")
                combined_text += self.summarizer.summarize(search_hit.text, 100, 80)[0]["summary_text"] + "\n"
                
            print(f"COMBINED TEXT: {combined_text}")
            
            # Sum
            result = self.summarizer.summarize(combined_text, 120, 100)[0]["summary_text"]

            print(f"FUNCTION OUTPUT: {result}")
            print("Using tool: search_arxiv")
            return result
        
        elif func_name == "notion":
            print(f"FUNCTION CALL: notion()")
            
            # Early out if not connected
            if not self.is_notion_connected:
                result = "You are not connected to notion. I cannot write to it."
                print(f"FUNCTION OUTPUT: {result}")
                print("Using tool: notion (failed - not connected)")
                return result
                
            
            self.notion.write_blocks(self.notion.conversation_to_notion_blocks(self.conversation_history[-10:]))

            result = "I have written the conversation to notion."
            print(f"FUNCTION OUTPUT: {result}")
            print("Using tool: notion")
            return result
        
        else:
            print(f"UNKNOWN FUNCTION: {func_name}")
            return f"Error: Unknown function '{func_name}'"
        
        
    def generate_response(self, user_text):
        # Get the last 5 turns of the conversation.
        self.conversation_history.append({"role":"user","content":user_text})
        
        # Frame the user's message to give context to the model.
        prompt = self.llm.tokenizer.apply_chat_template(
            [SYSTEM_PROMPT] + self.conversation_history[-10:], tokenize=False, add_generation_prompt=True
        )
        
        tok = self.llm.tokenizer
        
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        
        # End token
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [tok.eos_token_id] + ([im_end] if im_end is not None else [])
        
        out = self.llm(prompt,
                       max_new_tokens=160,
                       do_sample=False,
                       temperature=0.6,
                       top_p=0.9,
                       repetition_penalty=1.05,
                       return_full_text=False,
                       eos_token_id=eos_ids)
        
        # Get only the generated response
        bot = out[0]["generated_text"].strip()
        
        output = self.route_llm_output(bot)
        
        self.conversation_history.append({"role":"assistant","content":output})
        
        return output