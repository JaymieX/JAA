import os
import platform
from pydantic import BaseModel, TypeAdapter
from transformers import GenerationConfig
from enum import Enum
from typing import Optional, Type, TypedDict, Literal, Union, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
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
    router_decision:      Optional[RouterFunction]
    router_query:         Optional[str]
    query_keywords:       Optional[set[str]]
    vuln_response_dict:   Optional[dict]  # Store vulnerability check response
    final_response:       Optional[str]
    conversation_history: Annotated[list, add_messages]  # Use LangGraph's message reducer


class LLM:
    def __init__(self, profile : llm_loader.LLMProFile, notion_token, notion_page_id):
        # Remove self.conversation_history - now managed by LangGraph state

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
        self.notion     = Notion(os.environ["NOTION_API_KEY"], os.environ["NOTION_PAGE_ID"])
        
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

        # Compile with MemorySaver for conversation persistence
        self.checkpointer = MemorySaver()
        self.workflow = workflow.compile(checkpointer=self.checkpointer)
        
    
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

    def _messages_to_dicts(self, messages: list) -> list:
        """Convert LangChain message objects to plain dict format"""
        result = []
        for msg in messages:
            # Check if it's a LangChain message object
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # Convert LangChain message to dict
                role_map = {
                    'human': 'user',
                    'ai': 'assistant',
                    'system': 'system'
                }
                role = role_map.get(msg.type, 'user')
                result.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                # Already a dict, keep as is
                result.append(msg)
        return result
        
        
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


    def _router_node(self, state: AgentState) -> dict:
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
            query     = ""

        # Return only updates - LangGraph will merge into state
        return {
            "router_decision": func_name,
            "router_query":    query,
            "query_keywords":  keywords
        }


    def _route_decision(self, state: AgentState) -> str:
        """Routing function for conditional edges"""
        return state["router_decision"]


    def _search_arxiv_node(self, state: AgentState) -> dict:
        """Search ArXiv agent node"""
        query = state.get("router_query", "")
        print(f"FUNCTION CALL: search_arxiv({query})")

        if query == "":
            return {"final_response": "I am sorry I could not search the paper you need"}

        result = ""
        rag_results = self.rag_search.hybrid_search(query, 3)

        if len(rag_results) <= 0:
            return {"final_response": "I am sorry I could not search the paper you need"}

        # only sub summerize if we have more than 1 results
        if len(rag_results) > 1:
            for rag_result in rag_results:
                print(f"RAG result: {str(rag_result)} \n\n")
                result += self.summarizer.summarize(rag_result['text'], 180, 80)[0]["summary_text"] + "\n"

        result = self.summarizer.summarize(result, 200, 60)[0]["summary_text"]

        return {"final_response": result}


    def _notion_node(self, state: AgentState) -> dict:
        """Notion agent node"""
        print(f"FUNCTION CALL: notion()")

        if not self.is_notion_connected:
            result = "You are not connected to notion. I cannot write to it."
        else:
            self.notion.write_blocks(self.notion.conversation_to_notion_blocks(state.get("conversation_history", [])[-10:]))
            result = "I have written the conversation to notion."

        return {"final_response": result}


    def _human_text_node(self, state: AgentState) -> dict:
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

        # Convert LangChain messages back to dict format for LLM
        conversation = state.get("conversation_history", [])[-10:]
        conversation_dicts = self._messages_to_dicts(conversation)

        prompt = [llm_prompts.CHAT_SYSTEM_PROMPT] + conversation_dicts
        print(str(prompt))

        succeed, response = self._gen(prompt, gen_cfg)

        return {"final_response": response}


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


    def _check_vuln_node(self, state: AgentState) -> dict:
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
            return {
                "vuln_response_dict": response,
                "final_response": self._print_vlun_code_human(response)
            }
        else:
            return {"final_response": "I am sorry I could not help you with this, lets try something else."}

    def _vlun_relevence_node(self, state: AgentState) -> dict:
        """Add RAG-based relevance to vulnerability check response"""
        print(f"FUNCTION CALL: vuln_relevance()")

        vuln_dict = state.get("vuln_response_dict", {})

        # Use vulnerability_type as search keyword
        search_query = vuln_dict.get("vulnerability_type")

        # Get top 3 RAG results
        rag_results = self.rag_search.hybrid_search(search_query, top_k=3)

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
            return {"final_response": self._print_vlun_code_human(vuln_dict)}

        # If failed, return empty dict (no state updates)
        return {}


    def generate_response(self, user_text, thread_id: str = "default", stream: bool = False):
        """
        Main response generation using LangGraph workflow

        Args:
            user_text: User input message
            thread_id: Thread ID for conversation persistence (default: "default")
            stream: Whether to use streaming mode (default: False)

        Returns:
            Response text (or generator if stream=True)
        """
        # Config for thread-based persistence
        config = {"configurable": {"thread_id": thread_id}}

        # Check if user wants to clear conversation history
        if user_text.strip().lower() == "clear":
            # Get current state to access message IDs
            current_state = self.workflow.get_state(config)
            messages = current_state.values.get("conversation_history", [])

            # Create RemoveMessage for each message in history
            remove_messages = [RemoveMessage(id=msg.id) for msg in messages if hasattr(msg, 'id')]

            # Update state to remove all messages
            if remove_messages:
                self.workflow.update_state(
                    config,
                    {"conversation_history": remove_messages}
                )

            return ""

        # Create initial state - only pass required inputs
        initial_state = {
            "user_input": user_text,
            "conversation_history": [{"role": "user", "content": user_text}]
        }

        # Run LangGraph workflow
        if stream:
            # Streaming mode - return generator
            return self._stream_response(initial_state, config)
        else:
            # Standard invoke mode
            result = self.workflow.invoke(initial_state, config)
            output = result.get("final_response", "I'm sorry, I couldn't generate a response.")

            # Add assistant response to conversation history automatically
            # This ensures it's recorded regardless of which node was executed
            self.workflow.update_state(
                config,
                {"conversation_history": [{"role": "assistant", "content": output}]}
            )

            return output

    def _stream_response(self, initial_state, config):
        """Generator for streaming responses"""
        for event in self.workflow.stream(initial_state, config):
            # Yield intermediate results as they become available
            for node_name, node_output in event.items():
                if "final_response" in node_output:
                    yield node_output["final_response"]