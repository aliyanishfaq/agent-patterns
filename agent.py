"""
Two-Stage Agent Workflow with Review Mechanism

This module implements a generic two-stage agent workflow where:
1. First stage: An arbitrary ReAct agent produces initial output
2. Second stage: A review agent evaluates the output
3. If review is positive: workflow completes
4. If review is negative: retries the original agent with feedback

The implementation is designed to be generic and work with any ReAct agent
created using LangGraph's create_react_agent function.
"""

from typing import Annotated, Dict, Any, Literal, Optional, List
from typing_extensions import TypedDict
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent


@dataclass
class ReviewCriteria:
    """Configuration for review criteria and behavior."""
    max_retries: int = 3
    review_prompt: str = """
    Please review the following agent output and determine if it adequately addresses the user's request.
    
    User Request: {user_request}
    Agent Output: {agent_output}
    
    Consider:
    - Does the output directly answer the user's question?
    - Is the information accurate and complete?
    - Is the response clear and well-structured?
    
    Respond with either:
    - "APPROVED: [brief reason]" if the output is satisfactory
    - "REJECTED: [specific feedback for improvement]" if the output needs work
    """


class TwoStageAgentState(MessagesState):
    """
    Extended state for the two-stage agent workflow.
    
    Inherits from MessagesState to maintain conversation history,
    and adds fields to track workflow progress and review results.
    """
    # Current stage of the workflow
    current_stage: str = "initial"
    
    # Number of retry attempts made
    retry_count: int = 0
    
    # Review results and feedback
    review_result: Optional[str] = None
    review_feedback: Optional[str] = None
    
    # Store the initial user request for context
    initial_request: Optional[str] = None
    
    # Store the latest agent output for review
    latest_output: Optional[str] = None


class TwoStageAgentWorkflow:
    """
    A generic two-stage agent workflow that wraps any ReAct agent with a review mechanism.
    
    This class provides a framework for implementing agent workflows where an initial
    agent produces output that is then reviewed by a second agent. If the review is
    positive, the workflow completes. If negative, the original agent is retried with
    feedback until either success or max retries are reached.
    """
    
    def __init__(
        self,
        primary_agent,
        review_criteria: Optional[ReviewCriteria] = None,
        review_model: Optional[str] = None
    ):
        """
        Initialize the two-stage workflow.
        
        Args:
            primary_agent: A ReAct agent created with create_react_agent
            review_criteria: Configuration for review behavior
            review_model: Model to use for review agent (defaults to Claude)
        """
        self.primary_agent = primary_agent
        self.review_criteria = review_criteria or ReviewCriteria()
        
        # Initialize review model
        model_name = review_model or "claude-3-5-sonnet-20241022"
        self.review_model = ChatAnthropic(model=model_name)
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the StateGraph workflow for the two-stage agent."""
        
        # Create the graph with our custom state
        workflow = StateGraph(TwoStageAgentState)
        
        # Add nodes
        workflow.add_node("initial_agent", self._initial_agent_node)
        workflow.add_node("review_agent", self._review_agent_node)
        workflow.add_node("retry_agent", self._retry_agent_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.add_edge(START, "initial_agent")
        workflow.add_conditional_edges(
            "initial_agent",
            self._route_after_initial,
            {
                "review": "review_agent",
                "finalize": "finalize"
            }
        )
        workflow.add_conditional_edges(
            "review_agent",
            self._route_after_review,
            {
                "approved": "finalize",
                "retry": "retry_agent",
                "max_retries": "finalize"
            }
        )
        workflow.add_conditional_edges(
            "retry_agent",
            self._route_after_retry,
            {
                "review": "review_agent",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _initial_agent_node(self, state: TwoStageAgentState) -> Dict[str, Any]:
        """
        Execute the primary agent to generate initial output.
        
        Args:
            state: Current workflow state
            
        Returns:
            State update with agent output and stage information
        """
        # Extract the user request from the latest message
        if state["messages"]:
            latest_message = state["messages"][-1]
            if hasattr(latest_message, 'content'):
                user_request = latest_message.content
            else:
                user_request = str(latest_message)
        else:
            user_request = "No user request found"
        
        # Invoke the primary agent
        try:
            result = self.primary_agent.invoke({"messages": state["messages"]})
            
            # Extract the agent's response
            if result.get("messages"):
                agent_response = result["messages"][-1]
                if hasattr(agent_response, 'content'):
                    output_content = agent_response.content
                else:
                    output_content = str(agent_response)
            else:
                output_content = "No response generated"
            
            return {
                "messages": result.get("messages", []),
                "current_stage": "review_pending",
                "initial_request": user_request,
                "latest_output": output_content
            }
            
        except Exception as e:
            error_message = f"Error in initial agent: {str(e)}"
            return {
                "messages": [AIMessage(content=error_message)],
                "current_stage": "error",
                "latest_output": error_message
            }
    
    def _review_agent_node(self, state: TwoStageAgentState) -> Dict[str, Any]:
        """
        Execute the review agent to evaluate the primary agent's output.
        
        Args:
            state: Current workflow state
            
        Returns:
            State update with review results
        """
        try:
            # Prepare the review prompt
            review_prompt = self.review_criteria.review_prompt.format(
                user_request=state.get("initial_request", "Unknown request"),
                agent_output=state.get("latest_output", "No output")
            )
            
            # Get review from the model
            review_message = HumanMessage(content=review_prompt)
            review_response = self.review_model.invoke([review_message])
            
            review_content = review_response.content
            
            # Parse the review result
            if review_content.upper().startswith("APPROVED"):
                review_result = "approved"
                review_feedback = review_content
            elif review_content.upper().startswith("REJECTED"):
                review_result = "rejected"
                review_feedback = review_content
            else:
                # Default to rejected if format is unclear
                review_result = "rejected"
                review_feedback = f"REJECTED: Review format unclear. Original response: {review_content}"
            
            return {
                "current_stage": "review_complete",
                "review_result": review_result,
                "review_feedback": review_feedback
            }
            
        except Exception as e:
            return {
                "current_stage": "review_complete",
                "review_result": "rejected",
                "review_feedback": f"REJECTED: Error during review: {str(e)}"
            }
    
    def _retry_agent_node(self, state: TwoStageAgentState) -> Dict[str, Any]:
        """
        Execute the primary agent again with review feedback.
        
        Args:
            state: Current workflow state
            
        Returns:
            State update with retry attempt results
        """
        try:
            # Increment retry count
            new_retry_count = state.get("retry_count", 0) + 1
            
            # Prepare messages with feedback
            messages = list(state["messages"])
            
            # Add feedback message
            feedback_content = f"""
            The previous response needs improvement. Here's the feedback:
            
            {state.get('review_feedback', 'No specific feedback available')}
            
            Please provide an improved response that addresses these concerns.
            """
            
            messages.append(HumanMessage(content=feedback_content))
            
            # Invoke the primary agent with feedback
            result = self.primary_agent.invoke({"messages": messages})
            
            # Extract the agent's response
            if result.get("messages"):
                agent_response = result["messages"][-1]
                if hasattr(agent_response, 'content'):
                    output_content = agent_response.content
                else:
                    output_content = str(agent_response)
            else:
                output_content = "No response generated"
            
            return {
                "messages": result.get("messages", []),
                "current_stage": "retry_complete",
                "retry_count": new_retry_count,
                "latest_output": output_content
            }
            
        except Exception as e:
            error_message = f"Error in retry agent: {str(e)}"
            return {
                "messages": [AIMessage(content=error_message)],
                "current_stage": "error",
                "retry_count": state.get("retry_count", 0) + 1,
                "latest_output": error_message
            }
    
    def _finalize_node(self, state: TwoStageAgentState) -> Dict[str, Any]:
        """
        Finalize the workflow with appropriate completion message.
        
        Args:
            state: Current workflow state
            
        Returns:
            Final state update
        """
        # Determine completion reason
        if state.get("review_result") == "approved":
            completion_message = "✅ Response approved by review agent."
        elif state.get("retry_count", 0) >= self.review_criteria.max_retries:
            completion_message = f"⚠️ Maximum retries ({self.review_criteria.max_retries}) reached. Using last response."
        else:
            completion_message = "✅ Workflow completed."
        
        # Add completion message to conversation
        final_messages = list(state.get("messages", []))
        final_messages.append(AIMessage(content=completion_message))
        
        return {
            "messages": final_messages,
            "current_stage": "complete"
        }
    
    def _route_after_initial(self, state: TwoStageAgentState) -> str:
        """Route after initial agent execution."""
        if state.get("current_stage") == "error":
            return "finalize"
        return "review"
    
    def _route_after_review(self, state: TwoStageAgentState) -> str:
        """Route after review agent execution."""
        if state.get("review_result") == "approved":
            return "approved"
        elif state.get("retry_count", 0) >= self.review_criteria.max_retries:
            return "max_retries"
        else:
            return "retry"
    
    def _route_after_retry(self, state: TwoStageAgentState) -> str:
        """Route after retry agent execution."""
        if state.get("current_stage") == "error":
            return "finalize"
        return "review"
    
    def invoke(self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke the two-stage workflow.
        
        Args:
            input_data: Input data containing messages or other state
            config: Optional configuration for the workflow
            
        Returns:
            Final state after workflow completion
        """
        return self.graph.invoke(input_data, config=config)
    
    def stream(self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Stream the two-stage workflow execution.
        
        Args:
            input_data: Input data containing messages or other state
            config: Optional configuration for the workflow
            
        Yields:
            State updates during workflow execution
        """
        yield from self.graph.stream(input_data, config=config)


def create_two_stage_workflow(
    primary_agent,
    max_retries: int = 3,
    review_prompt: Optional[str] = None,
    review_model: Optional[str] = None
) -> TwoStageAgentWorkflow:
    """
    Factory function to create a two-stage agent workflow.
    
    This is a convenience function that wraps any ReAct agent with the
    two-stage review mechanism.
    
    Args:
        primary_agent: A ReAct agent created with create_react_agent
        max_retries: Maximum number of retry attempts (default: 3)
        review_prompt: Custom review prompt template (optional)
        review_model: Model to use for review agent (optional)
        
    Returns:
        TwoStageAgentWorkflow instance ready for use
        
    Example:
        ```python
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
        from langchain_anthropic import ChatAnthropic
        
        @tool
        def calculator(expression: str) -> str:
            "Calculate mathematical expressions"
            try:
                return str(eval(expression))
            except:
                return "Invalid expression"
        
        # Create a primary agent
        model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        primary_agent = create_react_agent(model, [calculator])
        
        # Wrap it in two-stage workflow
        workflow = create_two_stage_workflow(
            primary_agent=primary_agent,
            max_retries=2
        )
        
        # Use the workflow
        result = workflow.invoke({
            "messages": [{"role": "user", "content": "What is 15 * 23?"}]
        })
        ```
    """
    # Create review criteria
    criteria = ReviewCriteria(
        max_retries=max_retries,
        review_prompt=review_prompt or ReviewCriteria().review_prompt
    )
    
    return TwoStageAgentWorkflow(
        primary_agent=primary_agent,
        review_criteria=criteria,
        review_model=review_model
    )


# Export the compiled graph as 'app' for LangGraph platform compatibility
def create_default_workflow():
    """Create a default workflow for platform deployment."""
    from langchain_core.tools import tool
    
    @tool
    def echo_tool(message: str) -> str:
        """Echo the input message."""
        return f"Echo: {message}"
    
    # Create a simple default agent
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    default_agent = create_react_agent(model, [echo_tool])
    
    # Create the two-stage workflow
    workflow = create_two_stage_workflow(default_agent)
    
    return workflow.graph


# Export for LangGraph platform
app = create_default_workflow()
