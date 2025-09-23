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

class LLMProFile(Enum):
    SMALL = 0,
    LARGE = 1

# LangGraph State Definition
class AgentState(TypedDict):
    user_input: str
    router_decision: str
    final_response: str
    conversation_history: list
    

ROUTER_SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a prompt router with acess to tools. You must always respond in valid json.

    You have access to the following tools:
    1. search_arxiv(query) - for searching arXiv papers
    2. notion() - for sending content to Notion
    3. vulnerability_check(query) - for checking code vulnerability
    4. human_text(query) - for all other input

    When the user asks to search arXiv papers or research, respond with a JSON function call in this exact format:
    {"function": "search_arxiv", "arguments": {"query": "the search query"}}

    When the user asks to create a Notion note, respond with a JSON function call in this exact format:
    {"function": "notion"}
    
    When the user asks if a piece of code is vulnerable, or asks to fix a vulnerable code:
    {"function": "vulnerability_check", "arguments": {"query": "the query"}}

    For all other conversations, respond with a JSON function call in this exact format:
    {"function": "human_text", "arguments": {"query": "the human text"}}

    Examples:
    User: "Search for papers about quantum computing"
    Assistant: {"function": "search_arxiv", "arguments": {"query": "quantum computing"}}

    User: "Save this to Notion"
    Assistant: {"function": "notion"}
    
    User:
    Why is this code vulnerable?
    ```
    char buf[10];
    strcpy(buf, input);
    ```
    Assistant: {"function": "vulnerability_check", "arguments": {"query": "Why is this code vulnerable?  ```  char buf[10];  strcpy(buf, input);  ```"}}
    
    User:
    Why is this code vulnerable?
    ```
    <?php
    $query = "SELECT * FROM users WHERE name = \"" . $_GET["name"] . "\"";
    mysqli_query($conn, $query);
    ?>
    ```

    Assistant: {
    "function": "vulnerability_check",
        "arguments": {
            "query": "Why is this code vulnerable?  ```  <?php\\n    $query = \\\"SELECT * FROM users WHERE name = \\\\\\\"\\\" . $_GET[\\\"name\\\"] . \\\\\\\"\\\";\\n    mysqli_query($conn, $query);\\n    ?>  ```"
        }
    }

    User: "How are you today?"
    Assistant: {"function": "human_text", "arguments": {"query": "How are you today?"}}"""
}

CHAT_SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a helpful cyber security AI assistant. Try keeo your response under 100 words
    
    You must answer in a cute and weeby manner.
    Use words like kya~ and nyan~
    
    Examples:
    User: "Can you tell me what ddos is?"
    Assistant: Kya~ A DDoS is an attack that overwhelms a server with traffic to make it unavailable nyan!"""
}

SECURITY_SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a senior application security engineer.
    
    You must always respond in a strict 5-section text format, with numbered headings in order:

    1) Vulnerability Type
    2) Why It's Bad
    3) Exploit Scenario
    4) Evidence in Code
    5) Fix

    Rules:
    - Always output exactly 5 sections in this order.
    - Each section must be short, clear, and human-readable.
    - Evidence in Code: max 5 bullet points, each ≤120 chars, citing line numbers/patterns.
    - Fix: include both a short strategy and a minimal safe code snippet in the same programming language.
    - Wrap both User inputs and Assistant outputs in ``` ... ```.
    - Do not output JSON or any extra text outside the 5 sections.

    Examples:

    User:
    Why is this code vulnerable?
    ```
    char buf[10];
    strcpy(buf, input);
    ```

    Assistant:
    1) Vulnerability Type
    CWE-120: Buffer Copy without Checking Size

    2) Why It's Bad
    Allows memory overwrite, leading to crashes or code execution.

    3) Exploit Scenario
    An attacker inputs more than 10 characters to overflow the stack buffer.

    4) Evidence in Code
    - Line 2: `strcpy(buf, input);` → unbounded copy into fixed-size buffer

    5) Fix
    Strategy: Use bounded copy with explicit size limit.
    Patched Code:
    ```
    strncpy(buf, input, sizeof(buf)-1);
    buf[sizeof(buf)-1] = '\\0';
    ```"""
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

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "search_arxiv": "search_arxiv",
                "notion": "notion",
                "human_text": "human_text"
            }
        )

        # All nodes go to END
        workflow.add_edge("search_arxiv", END)
        workflow.add_edge("notion", END)
        workflow.add_edge("human_text", END)

        self.workflow = workflow.compile()

    def _router_node(self, state: AgentState) -> AgentState:
        """Router node that decides which agent to use"""
        prompt = self.llm.tokenizer.apply_chat_template(
            [ROUTER_SYSTEM_PROMPT, {"role":"user","content":state["user_input"]}],
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
            decision = json.loads(router_output)
            func_name = decision.get("function", "human_text")
        except (json.JSONDecodeError, TypeError):
            func_name = "human_text"

        state["router_decision"] = func_name
        return state

    def _route_decision(self, state: AgentState) -> Literal["search_arxiv", "notion", "human_text"]:
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
            [CHAT_SYSTEM_PROMPT] + state["conversation_history"][-10:],
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

    def router(self, user_text):
        """Use LangGraph workflow for routing (deprecated - kept for compatibility)"""
        return self.generate_response(user_text)
        
        
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
            print(f"ROUTING ERROR OUTPUT: {llm_output}")
            return "I am sorry, I could not help with this. Can you try something else?"

        if func_name == "search_arxiv":
            query = args.get("query", "")
            print(f"FUNCTION CALL: search_arxiv(query='{query}')")
            
            """
            # Get top x rag paper and combine
            rag_results = self.rag_search.hybrid_search(query, 3)
            combined_text = ""
            for search_hit in rag_results["hits"]:
                print(f"HIT TEXT: {search_hit.text}")
                combined_text += self.summarizer.summarize(search_hit.text, 100, 80)[0]["summary_text"] + "\n"
                
            print(f"COMBINED TEXT: {combined_text}")
            
            # Sum
            result = self.summarizer.summarize(combined_text, 120, 100)[0]["summary_text"]
            """
            
            result = "Function disabled until future"

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
        
        elif func_name == "vulnerability_check":
            print(f"FUNCTION CALL: vulnerability_check()")
            query = args.get("query", "")
            
            return self.generate_structured_response(query)
        
        elif func_name == "human_text":
            print(f"FUNCTION CALL: human_text()")
            query = args.get("query", "")
            
            return self.generate_response(query)
        
        else:
            print(f"UNKNOWN FUNCTION: {func_name}")
            return f"Error: Unknown function '{func_name}'"
        
        
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
            self.conversation_history.append({"role":"assistant","content":fallback_response})
            return fallback_response
    
    
    def generate_structured_response(self, user_text):
        # Get the last 5 turns of the conversation.
        self.conversation_history.append({"role":"user","content":user_text})
        
        # Construct prompt.
        prompt = self.llm.tokenizer.apply_chat_template(
            [SECURITY_SYSTEM_PROMPT, {"role":"user","content":user_text}], tokenize=False, add_generation_prompt=True
        )
        
        tok = self.llm.tokenizer
        
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        
        # End token
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
        
        # Get only the generated response
        output = out[0]["generated_text"].strip()
        
        self.conversation_history.append({"role":"assistant","content":output})
        
        return output