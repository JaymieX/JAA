import json
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig, pipeline, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from enum import Enum
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

import llm_loader
from summarize import Summarizer
from notion import Notion
from rag.rag_engine import RAGEngine
import llm_prompts

# LangGraph State Definition
class AgentState(TypedDict):
    user_input: str
    router_decision: str
    final_response: str
    conversation_history: list


class LLM:
    def __init__(self, profile : llm_loader.LLMProFile, notion_token, notion_page_id):
        self.conversation_history = []

        self.llm = llm_loader.load_llm(profile)
        if (self.llm is not None):    
            print("LLM loaded.")
        else:
            print("LLM fail to loaded.")
            return
        
        try:
            self.rag_search = RAGEngine(collection_name="rag_collection")
            print("RAG Engine loaded.")
        except Exception as e:
            print(f"RAG Engine failed to load: {e}")
            self.rag_search = None
            
        self.summarizer = Summarizer()
        self.notion     = Notion(notion_token, notion_page_id)
        
        self.is_notion_connected = self.notion.is_connected()
        
        # EOS ids
        tok = self.llm.tokenizer
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token

        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = [tok.eos_token_id] + ([im_end] if im_end is not None else [])

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
        
        
    def _gen(self, messages, gen_cfg : GenerationConfig):
        prompt = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True
        )
        
        out = self.llm(prompt, generation_config=gen_cfg, return_full_text=False)
        return out[0]["generated_text"].strip()


    def _router_node(self, state: AgentState) -> AgentState:
        """Router node that decides which agent to use"""
        gen_cfg = GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=self.eos_ids
        )

        router_output = self._gen([llm_prompts.ROUTER_SYSTEM_PROMPT, {"role":"user","content":state["user_input"]}], gen_cfg)

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

        gen_cfg = GenerationConfig(
            max_new_tokens=160,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=self.eos_ids
        )

        response = self._gen([llm_prompts.CHAT_SYSTEM_PROMPT] + state["conversation_history"][-10:], gen_cfg)

        state["final_response"] = response
        return state


    def _check_vuln_node(self, state: AgentState) -> AgentState:
        """Vulnerability check agent node"""
        print(f"FUNCTION CALL: vulnerability_check()")

        gen_cfg = GenerationConfig(
            max_new_tokens=400,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=self.eos_ids
        )

        response = self._gen([llm_prompts.SECURITY_SYSTEM_PROMPT, {"role":"user","content":state["user_input"]}], gen_cfg)

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