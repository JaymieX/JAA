import json
import platform
from pathlib import Path
from pydantic import BaseModel
import torch
from transformers import BitsAndBytesConfig, pipeline, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from enum import Enum
from typing import Optional, Type, TypedDict, Literal, Union
from langgraph.graph import StateGraph, END

# Use vLLM on Unix systems, transformers on Windows
if platform.system() == 'Windows':
    import llm_loader_transformer as llm_loader
else:
    print("Using vllm for faster inference")
    import llm_loader_vllm as llm_loader

from summarize import Summarizer
from notion import Notion
from rag.rag_engine import RAGEngine
import llm_prompts

# Router Function Enum
class RouterFunction(str, Enum):
    SEARCH_ARXIV        = "search_arxiv"
    NOTION              = "notion"
    HUMAN_TEXT          = "human_text"
    VULNERABILITY_CHECK = "vulnerability_check"

# LangGraph State Definition
class AgentState(TypedDict):
    user_input:           str
    router_decision:      RouterFunction
    router_query:         str
    final_response:       str
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
            self.rag_search = RAGEngine(collection_name="demo_collection")
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
        workflow.add_node("router",                                 self._router_node)
        workflow.add_node(RouterFunction.SEARCH_ARXIV.value,        self._search_arxiv_node)
        workflow.add_node(RouterFunction.NOTION.value,              self._notion_node)
        workflow.add_node(RouterFunction.HUMAN_TEXT.value,          self._human_text_node)
        workflow.add_node(RouterFunction.VULNERABILITY_CHECK.value, self._check_vuln_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router (using enum values)
        mapping = {r.value: r.value for r in RouterFunction}
        
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            mapping
        )

        # All nodes go to END
        workflow.add_edge(RouterFunction.SEARCH_ARXIV.value,        END)
        workflow.add_edge(RouterFunction.NOTION.value,              END)
        workflow.add_edge(RouterFunction.HUMAN_TEXT.value,          END)
        workflow.add_edge(RouterFunction.VULNERABILITY_CHECK.value, END)

        self.workflow = workflow.compile()
        
        
    def _gen(self, messages, gen_cfg : GenerationConfig, structured_output_schema: Optional[Union[Type[BaseModel], type]] = None):
        prompt = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True
        )
        
        out = self.llm(prompt, generation_config=gen_cfg, return_full_text=False, structured_output_schema=structured_output_schema)
        return out[0]["generated_text"].strip()


    def _router_node(self, state: AgentState) -> AgentState:
        """Router node that decides which agent to use"""
        gen_cfg = GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=self.eos_ids
        )

        router_output = self._gen([llm_prompts.ROUTER_SYSTEM_PROMPT, {"role":"user","content":state["user_input"]}], gen_cfg, llm_prompts.ROUTER_RESPONSE_JSON_ENFORCE)

        try:
            # Attempt to extract function name if we failed default to human_text
            decision  = json.loads(router_output)
            func_name = decision.get("function", RouterFunction.HUMAN_TEXT)
            query     = decision.get("query", "")

            # Validate against enum
            valid_routes = {route.value for route in RouterFunction}
            if func_name not in valid_routes:
                func_name = RouterFunction.HUMAN_TEXT
                query     = ""

        except (json.JSONDecodeError, TypeError):
            func_name = RouterFunction.HUMAN_TEXT

        state["router_decision"] = func_name
        state["router_query"]    = query
        
        return state


    def _route_decision(self, state: AgentState) -> str:
        """Routing function for conditional edges"""
        return state["router_decision"]


    def _search_arxiv_node(self, state: AgentState) -> AgentState:
        """Search ArXiv agent node"""
        query = state["router_query"]
        print(f"FUNCTION CALL: search_arxiv({query})")
        
        if query == "":
            state["final_response"] = "I am sorry I could not search the paper you need"
            return state
        
        result = ""
        rag_results = self.rag_search.hybrid_search(query, 3)
        
        if len(rag_results) <= 0:
            state["final_response"] = "I am sorry I could not search the paper you need"
            return state
        
        # only sub summerize if we have more than 1 results
        if len(rag_results) > 1:
            for rag_result in rag_results:
                result += self.summarizer.summarize(rag_result['text'], 180, 140)[0]["summary_text"] + "\n"
            
        result = self.summarizer.summarize(result, 200, 160)[0]["summary_text"]
        
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
            router_decision=RouterFunction.HUMAN_TEXT,
            final_response="",
            json_response="",
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