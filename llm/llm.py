from unsloth import FastLanguageModel
import json
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig, pipeline, GenerationConfig
from peft import PeftModel
from enum import Enum
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from summarize import Summarizer
from notion import Notion
#from search import Search
import llm_prompts

class LLMProFile(Enum):
    SMALL = 0,
    LARGE = 1

# LangGraph State Definition
class AgentState(TypedDict):
    user_input: str
    router_decision: str
    final_response: str
    conversation_history: list


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
            ft_base_model, ft_tokenizer = FastLanguageModel.from_pretrained(
                "unsloth/Qwen2.5-7B-Instruct", #7B
                dtype=None,
                max_seq_length=2048,
                device_map="auto"
            )

            # Load LoRA adapter
            ft_model = PeftModel.from_pretrained(ft_base_model, "Amie69/Qwen2.5-7B-cve-coder")

            # Verify LoRA adapter is loaded
            if hasattr(ft_model, 'peft_config') and ft_model.peft_config:
                print(f"✓ LoRA adapter loaded successfully: {list(ft_model.peft_config.keys())}")
            else:
                print("⚠️ Warning: LoRA adapter may not be loaded correctly")

            # Optimize for inference (2x speedup)
            ft_model = FastLanguageModel.for_inference(ft_model)
            ft_model.eval() # Inference mode

            self.llm = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)
            
        else:
            print("LLM fail to load.")
            return
            
        
        print("LLM loaded.")
        
        #self.rag_search = Search()
        self.summarizer = Summarizer()
        self.notion     = Notion(notion_token, notion_page_id)
        
        self.is_notion_connected = self.notion.is_connected()

        # Initialize LangGraph workflow
        self._setup_langgraph_workflow()


    def _setup_langgraph_workflow(self):
        """Setup LangGraph workflow for agent routing"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("search_arxiv", self._search_arxiv_node)
        workflow.add_node("notion", self._notion_node)
        workflow.add_node("human_text", self._human_text_node)
        workflow.add_node("vulnerability_check", self._check_vuln_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "search_arxiv": "search_arxiv",
                "notion": "notion",
                "human_text": "human_text",
                "vulnerability_check": "vulnerability_check"
            }
        )

        # All nodes go to END
        workflow.add_edge("search_arxiv", END)
        workflow.add_edge("notion", END)
        workflow.add_edge("human_text", END)
        workflow.add_edge("vulnerability_check", END)

        self.workflow = workflow.compile()


    def _router_node(self, state: AgentState) -> AgentState:
        """Router node that decides which agent to use"""
        prompt = self.llm.tokenizer.apply_chat_template(
            [llm_prompts.ROUTER_SYSTEM_PROMPT, {"role":"user","content":state["user_input"]}],
            tokenize=False, add_generation_prompt=True
        )

        tok = self.llm.tokenizer
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token

        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [tok.eos_token_id] + ([im_end] if im_end is not None else [])

        gen_cfg = GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_ids
        )

        out = self.llm(prompt, generation_config=gen_cfg, return_full_text=False)
        router_output = out[0]["generated_text"].strip()

        try:
            # Attempt to extract function name if we failed default to human_text
            decision = json.loads(router_output)
            func_name = decision.get("function", "human_text")
        except (json.JSONDecodeError, TypeError):
            func_name = "human_text"

        state["router_decision"] = func_name
        
        return state


    def _route_decision(self, state: AgentState) -> Literal["search_arxiv", "notion", "human_text", "vulnerability_check"]:
        """Routing function for conditional edges"""
        return state["router_decision"]


    def _search_arxiv_node(self, state: AgentState) -> AgentState:
        """Search ArXiv agent node"""
        print(f"FUNCTION CALL: search_arxiv")
        result = "Function disabled until future"
        state["final_response"] = result
        return state


    def _notion_node(self, state: AgentState) -> AgentState:
        """Notion agent node"""
        print(f"FUNCTION CALL: notion()")

        if not self.is_notion_connected:
            result = "You are not connected to notion. I cannot write to it."
        else:
            self.notion.write_blocks(self.notion.conversation_to_notion_blocks(state["conversation_history"][-10:]))
            result = "I have written the conversation to notion."

        state["final_response"] = result
        return state


    def _human_text_node(self, state: AgentState) -> AgentState:
        """Human text conversation agent node"""
        print(f"FUNCTION CALL: human_text()")

        # Use existing generate_response logic
        prompt = self.llm.tokenizer.apply_chat_template(
            [llm_prompts.CHAT_SYSTEM_PROMPT] + state["conversation_history"][-10:],
            tokenize=False, add_generation_prompt=True
        )

        tok = self.llm.tokenizer
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token

        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [tok.eos_token_id] + ([im_end] if im_end is not None else [])

        gen_cfg = GenerationConfig(
            max_new_tokens=160,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=eos_ids
        )

        out = self.llm(prompt, generation_config=gen_cfg, return_full_text=False)
        response = out[0]["generated_text"].strip()

        state["final_response"] = response
        return state


    def _check_vuln_node(self, state: AgentState) -> AgentState:
        """Vulnerability check agent node"""
        print(f"FUNCTION CALL: vulnerability_check()")

        # Construct prompt
        prompt = self.llm.tokenizer.apply_chat_template(
            [llm_prompts.SECURITY_SYSTEM_PROMPT, {"role":"user","content":state["user_input"]}],
            tokenize=False, add_generation_prompt=True
        )

        tok = self.llm.tokenizer
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token

        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [tok.eos_token_id] + ([im_end] if im_end is not None else [])

        gen_cfg = GenerationConfig(
            max_new_tokens=400,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=eos_ids
        )

        out = self.llm(prompt, generation_config=gen_cfg, return_full_text=False)
        response = out[0]["generated_text"].strip()

        state["final_response"] = response
        return state
        
        
    def generate_response(self, user_text):
        """Main response generation using LangGraph workflow"""
        # Update conversation history
        self.conversation_history.append({"role":"user","content":user_text})

        # Create initial state
        initial_state = AgentState(
            user_input=user_text,
            router_decision="",
            final_response="",
            conversation_history=self.conversation_history.copy()
        )

        # Run LangGraph workflow
        try:
            result = self.workflow.invoke(initial_state)
            output = result["final_response"]

            # Update conversation history with response
            self.conversation_history.append({"role":"assistant","content":output})

            return output

        except Exception as e:
            print(f"LangGraph workflow error: {e}")
            # Fallback to simple response
            fallback_response = "I'm sorry, I encountered an error processing your request."
            
            return fallback_response