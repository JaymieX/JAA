import json
import platform
from pathlib import Path
from pydantic import BaseModel, TypeAdapter
import torch
from transformers import BitsAndBytesConfig, pipeline, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from enum import Enum
from typing import Optional, Type, TypedDict, Literal, Union
from langgraph.graph import StateGraph, END
from json_repair import repair_json


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
    query_keywords:       set[str]
    vuln_response_dict:   dict  # Store vulnerability check response
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
        workflow.add_node("vuln_relevance",                         self._vlun_relevence_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router (using enum values)
        mapping = {r.value: r.value for r in RouterFunction}
        
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            mapping
        )

        # All nodes go to END except vulnerability_check (conditional)
        workflow.add_edge(RouterFunction.SEARCH_ARXIV.value,        END)
        workflow.add_edge(RouterFunction.NOTION.value,              END)
        workflow.add_edge(RouterFunction.HUMAN_TEXT.value,          END)
        workflow.add_edge("vuln_relevance",                         END)

        # Conditional edge from vulnerability_check
        workflow.add_conditional_edges(
            RouterFunction.VULNERABILITY_CHECK.value,
            self._check_rag_keyword,
            {
                "vuln_relevance": "vuln_relevance",
                END: END
            }
        )

        self.workflow = workflow.compile()
        
    
    def _check_rag_keyword(self, state: AgentState) -> Literal["vuln_relevance", "END"]:
        """Check if 'rag' keyword is present to route to relevance node"""
        if 'rag' in state.get("query_keywords", set()):
            return "vuln_relevance"
        return END

    def _extract_keywords(self, text: str, keywords: set[str] = {'rag'}) -> set[str]:
        """Extract keywords found in text (case-insensitive)"""
        text_lower = text.lower()
        found = set()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found.add(keyword)
        return found
        
        
    def _gen(self, messages, gen_cfg : GenerationConfig, structured_output_schema: Optional[Union[Type[BaseModel], type]] = None):
        prompt = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True
        )

        out = self.llm(prompt, generation_config=gen_cfg, return_full_text=False, structured_output_schema=structured_output_schema)
        generated_text = out[0]["generated_text"].strip()

        # If structured output is requested, repair and validate JSON
        if structured_output_schema is not None:
            try:
                # Repair JSON and get object directly (faster)
                generated_obj = repair_json(generated_text, return_objects=True)

                # Validate with Pydantic schema
                if hasattr(structured_output_schema, '__origin__'):
                    # Union type - try to parse with TypeAdapter
                    adapter = TypeAdapter(structured_output_schema)
                    adapter.validate_python(generated_obj)
                    return True, generated_obj
                else:
                    # Regular Pydantic model
                    structured_output_schema.model_validate(generated_obj)
                    return True, generated_obj

            except Exception as e:
                # Validation failed, return failure with raw text
                print(f"⚠️ JSON validation failed: {e}")
                return False, generated_text

        # Normal text generation, always succeed
        return True, generated_text


    def _router_node(self, state: AgentState) -> AgentState:
        """Router node that decides which agent to use"""
        gen_cfg = GenerationConfig(
            max_new_tokens=50,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=self.eos_ids
        )
        
        keywords = self._extract_keywords(state["user_input"])

        succeed, gen_obj = self._gen([llm_prompts.ROUTER_SYSTEM_PROMPT, {"role":"user","content":state["user_input"]}], gen_cfg, llm_prompts.ROUTER_RESPONSE_JSON_ENFORCE)

        if succeed:
            # Attempt to extract function name if we failed default to human_text
            func_name = gen_obj.get("function", RouterFunction.HUMAN_TEXT)
            query     = gen_obj.get("query", "")

            # Validate against enum
            valid_routes = {route.value for route in RouterFunction}
            if func_name not in valid_routes:
                func_name = RouterFunction.HUMAN_TEXT
                query     = ""

        else:
            func_name = RouterFunction.HUMAN_TEXT

        state["router_decision"] = func_name
        state["router_query"]    = query
        state["query_keywords"]  = keywords
        
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
                print(f"RAG result: {str(rag_result)} \n\n")
                result += self.summarizer.summarize(rag_result['text'], 180, 80)[0]["summary_text"] + "\n"
            
        result = self.summarizer.summarize(result, 200, 60)[0]["summary_text"]
        
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

        succeed, response = self._gen([llm_prompts.CHAT_SYSTEM_PROMPT] + state["conversation_history"][-10:], gen_cfg)

        state["final_response"] = response
        return state


    def _print_vlun_code_human(self, response):
        # Construct human readable 5 points from validated dictionary
        evidence_lines = []
        for evidence in response.get("evidence_in_code", []):
            evidence_lines.append(
                f"- Line {evidence.get('line_number', 'N/A')}: "
                f"`{evidence.get('evidence_code', '')}` → {evidence.get('explanation', '')}"
            )

        fix_block = response.get("fix", {})

        # Build response sections with proper indentation
        sections = [
            "1) Vulnerability Type",
            response.get('vulnerability_type', 'Unknown'),
            "",
            "2) Why It's Bad",
            response.get('why_its_bad', 'N/A'),
            "",
            "3) Exploit Scenario",
            response.get('exploit_scenario', 'N/A'),
            "",
            "4) Evidence in Code",
            '\n'.join(evidence_lines),
            "",
            "5) Fix",
            f"Strategy: {fix_block.get('strategy', 'N/A')}",
            "Patched Code:",
            f"```{response.get('code_language', '')}",
            fix_block.get('patched_code', ''),
            "```"
        ]

        # Add optional 6) Relevance section if present
        if 'relevance' in response:
            sections.extend([
                "",
                "6) Relevance",
                response.get('relevance')
            ])

        return '\n'.join(sections)


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

        succeed, response = self._gen([llm_prompts.SECURITY_SYSTEM_PROMPT, {"role":"user","content":state["user_input"]}], gen_cfg, llm_prompts.VlunCheckResponse)

        if succeed:
            state["vuln_response_dict"] = response
            state["final_response"] = self._print_vlun_code_human(response)
        else:
            state["final_response"] = "I am sorry I could not help you with this, lets try something else."

        return state

    def _vlun_relevence_node(self, state: AgentState) -> AgentState:
        """Add RAG-based relevance to vulnerability check response"""
        print(f"FUNCTION CALL: vuln_relevance()")

        vuln_dict = state.get("vuln_response_dict", {})

        # Use vulnerability_type as search keyword
        search_query = vuln_dict.get("vulnerability_type")

        # Get top 3 RAG results
        rag_results = self.rag_search.hybrid_search(search_query, top_k=3)

        # Early out if no results found
        if not rag_results:
            return state

        # Build context from RAG results
        rag_context = "\n\n".join([
            f"Reference {i+1}: {result['text']}"
            for i, result in enumerate(rag_results)
        ])

        # Generate relevance using RAG context
        gen_cfg = GenerationConfig(
            max_new_tokens=200,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=self.eos_ids
        )

        # Build prompt with proper indentation
        prompt_parts = [
            "Using these references, explain the relevance and context of this vulnerability:",
            "",
            rag_context,
            "",
            f"Vulnerability Type: {vuln_dict.get('vulnerability_type', 'Unknown')}"
        ]
        prompt = '\n'.join(prompt_parts)

        succeed, relevance_text = self._gen(
            [llm_prompts.SECURITY_RAG_SYSTEM_PROMPT, {"role":"user","content":prompt}],
            gen_cfg
        )

        if succeed:
            # Add relevance to vuln dict and regenerate response
            vuln_dict["relevance"] = relevance_text
            state["final_response"] = self._print_vlun_code_human(vuln_dict)

        return state


    def generate_response(self, user_text):
        """Main response generation using LangGraph workflow"""
        # Update conversation history
        self.conversation_history.append({"role":"user","content":user_text})

        # Create initial state
        initial_state = AgentState(
            user_input=user_text,
            router_decision=RouterFunction.HUMAN_TEXT,
            router_query="",
            query_keywords={},
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