"""
Two-Stage Agent Workflow with Review Mechanism

This module implements a generic two-stage agent workflow where:
1. An initial ReAct agent processes the user's request
2. A review agent evaluates the initial output
3. If the review passes, the workflow finishes
4. If the review fails, the request is sent back to the initial agent for improvement

The implementation is generic and accepts any ReAct agent creation function.
"""

from typing import Callable, Dict, Any, List, Literal, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command


class TwoStageState(TypedDict):
    """State schema for the two-stage review workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    initial_output: str
    review_result: str
    review_passed: bool
    iteration_count: int
    max_iterations: int


class TwoStageReviewAgent:
    """
    A generic two-stage agent workflow that accepts any ReAct agent creation function.
    
    The workflow consists of:
    1. Initial agent: Processes the user's request using the provided ReAct agent
    2. Review agent: Evaluates the initial output for quality and completeness
    3. Decision logic: Routes based on review results (pass/fail)
    
    If the review fails, the workflow loops back to the initial agent with feedback.
    """
    
    def __init__(
        self,
        agent_creator: Callable[..., Any],
        agent_tools: List[Any] = None,
        model_name: str = "claude-3-5-sonnet-20241022",
        max_iterations: int = 3
    ):
        """
        Initialize the TwoStageReviewAgent.
        
        Args:
            agent_creator: Function to create the initial ReAct agent (e.g., create_react_agent)
            agent_tools: List of tools to provide to the initial agent
            model_name: Name of the model to use for both agents
            max_iterations: Maximum number of review iterations before stopping
        """
        self.agent_creator = agent_creator
        self.agent_tools = agent_tools or []
        self.model_name = model_name
        self.max_iterations = max_iterations
        
        # Initialize the chat model for the review agent
        self.review_model = ChatAnthropic(model=model_name)
        
        # Create the initial ReAct agent
        self.initial_agent = self._create_initial_agent()
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _create_initial_agent(self):
        """Create the initial ReAct agent using the provided creator function."""
        return self.agent_creator(
            model=ChatAnthropic(model=self.model_name),
            tools=self.agent_tools
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the StateGraph for the two-stage workflow."""
        builder = StateGraph(TwoStageState)
        
        # Add nodes
        builder.add_node("initial_agent", self._initial_agent_node)
        builder.add_node("review_agent", self._review_agent_node)
        builder.add_node("prepare_retry", self._prepare_retry_node)
        
        # Add edges
        builder.add_edge(START, "initial_agent")
        builder.add_conditional_edges(
            "initial_agent",
            self._should_review,
            {
                "review": "review_agent",
                "end": END
            }
        )
        builder.add_conditional_edges(
            "review_agent",
            self._review_decision,
            {
                "pass": END,
                "retry": "prepare_retry",
                "max_iterations": END
            }
        )
        builder.add_edge("prepare_retry", "initial_agent")
        
        return builder.compile()
    
    def _initial_agent_node(self, state: TwoStageState) -> Dict[str, Any]:
        """
        Node that runs the initial ReAct agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with initial agent output
        """
        # Extract the original query from messages if not already set
        if not state.get("original_query"):
            # Find the first human message as the original query
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break
            else:
                original_query = "No query found"
        else:
            original_query = state["original_query"]
        
        # Prepare messages for the initial agent
        if state.get("iteration_count", 0) > 0:
            # This is a retry - include review feedback
            messages = [
                HumanMessage(content=original_query),
                SystemMessage(content=f"Previous attempt feedback: {state.get('review_result', '')}")
            ]
        else:
            # First attempt - use original messages
            messages = state["messages"]
        
        # Run the initial agent
        try:
            result = self.initial_agent.invoke({"messages": messages})
            
            # Extract the final AI message content as the initial output
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            initial_output = ai_messages[-1].content if ai_messages else "No output generated"
            
            return {
                "original_query": original_query,
                "initial_output": initial_output,
                "messages": result["messages"]
            }
        except Exception as e:
            return {
                "original_query": original_query,
                "initial_output": f"Error in initial agent: {str(e)}",
                "messages": state["messages"] + [AIMessage(content=f"Error: {str(e)}")]
            }
    
    def _review_agent_node(self, state: TwoStageState) -> Dict[str, Any]:
        """
        Node that runs the review agent to evaluate the initial output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with review results
        """
        review_prompt = f"""
        You are a quality review agent. Your job is to evaluate whether the following response adequately addresses the user's query.

        Original Query: {state['original_query']}
        
        Response to Review: {state['initial_output']}
        
        Please evaluate the response based on:
        1. Completeness: Does it fully address the query?
        2. Accuracy: Is the information correct?
        3. Clarity: Is it well-structured and easy to understand?
        4. Relevance: Does it stay on topic?
        
        Respond with either:
        - "PASS: [brief explanation]" if the response is satisfactory
        - "FAIL: [specific feedback for improvement]" if the response needs work
        
        Be constructive in your feedback and specific about what needs improvement.
        """
        
        try:
            review_message = HumanMessage(content=review_prompt)
            review_result = self.review_model.invoke([review_message])
            review_content = review_result.content
            
            # Determine if review passed
            review_passed = review_content.strip().upper().startswith("PASS")
            
            return {
                "review_result": review_content,
                "review_passed": review_passed
            }
        except Exception as e:
            return {
                "review_result": f"Error in review: {str(e)}",
                "review_passed": False
            }
    
    def _prepare_retry_node(self, state: TwoStageState) -> Dict[str, Any]:
        """
        Node that prepares the state for a retry iteration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with incremented iteration count
        """
        current_iteration = state.get("iteration_count", 0)
        return {
            "iteration_count": current_iteration + 1,
            "max_iterations": self.max_iterations
        }
    
    def _should_review(self, state: TwoStageState) -> Literal["review", "end"]:
        """
        Conditional edge function to determine if we should proceed to review.
        
        Args:
            state: Current workflow state
            
        Returns:
            "review" to proceed to review, "end" to terminate
        """
        # Always review unless there was an error in the initial agent
        if state.get("initial_output", "").startswith("Error"):
            return "end"
        return "review"
    
    def _review_decision(self, state: TwoStageState) -> Literal["pass", "retry", "max_iterations"]:
        """
        Conditional edge function to determine next step after review.
        
        Args:
            state: Current workflow state
            
        Returns:
            "pass" if review passed, "retry" to try again, "max_iterations" if limit reached
        """
        # Check if we've reached max iterations
        current_iteration = state.get("iteration_count", 0)
        if current_iteration >= self.max_iterations:
            return "max_iterations"
        
        # Check review result
        if state.get("review_passed", False):
            return "pass"
        else:
            return "retry"
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the two-stage workflow.
        
        Args:
            input_data: Input containing messages or query
            
        Returns:
            Final workflow state
        """
        # Ensure we have messages in the correct format
        if "messages" not in input_data:
            if "query" in input_data:
                input_data["messages"] = [HumanMessage(content=input_data["query"])]
            else:
                raise ValueError("Input must contain either 'messages' or 'query'")
        
        # Initialize state
        initial_state = {
            "messages": input_data["messages"],
            "original_query": "",
            "initial_output": "",
            "review_result": "",
            "review_passed": False,
            "iteration_count": 0,
            "max_iterations": self.max_iterations
        }
        
        # Run the workflow
        return self.graph.invoke(initial_state)


# Example tools for demonstration
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        Result of the calculation
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text and provide basic statistics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Analysis results including word count, character count, etc.
    """
    if not text:
        return "Error: No text provided"
    
    words = text.split()
    sentences = text.split('.')
    
    analysis = {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    
    return f"Text Analysis: {analysis}"


# Create the default two-stage agent with example tools
def create_default_two_stage_agent() -> TwoStageReviewAgent:
    """Create a default two-stage agent with basic tools."""
    return TwoStageReviewAgent(
        agent_creator=create_react_agent,
        agent_tools=[calculator, text_analyzer],
        model_name="claude-3-5-sonnet-20241022",
        max_iterations=3
    )


# Export the compiled graph as 'app' for LangGraph Platform compatibility
app = create_default_two_stage_agent().graph
