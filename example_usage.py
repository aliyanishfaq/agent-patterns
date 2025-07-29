"""
Example Usage of Two-Stage Agent Workflow

This module demonstrates how to use the TwoStageAgentWorkflow with various
scenarios including successful completion, retry loops, and different
configuration options.
"""

import re
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from agent import create_two_stage_workflow, ReviewCriteria


# Sample tools for demonstration
@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Basic safety check - only allow numbers, operators, and parentheses
        if not re.match(r'^[0-9+\-*/().\s]+$', expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."
        
        # Evaluate the expression
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


@tool
def text_processor(text: str, operation: str = "uppercase") -> str:
    """
    Process text with various operations.
    
    Args:
        text: The text to process
        operation: The operation to perform. Options: "uppercase", "lowercase", "reverse", "word_count"
        
    Returns:
        The processed text result
    """
    try:
        if operation == "uppercase":
            return f"Uppercase result: {text.upper()}"
        elif operation == "lowercase":
            return f"Lowercase result: {text.lower()}"
        elif operation == "reverse":
            return f"Reversed text: {text[::-1]}"
        elif operation == "word_count":
            word_count = len(text.split())
            return f"Word count: {word_count} words in '{text}'"
        else:
            return f"Unknown operation '{operation}'. Available operations: uppercase, lowercase, reverse, word_count"
    except Exception as e:
        return f"Error processing text: {str(e)}"


@tool
def weather_info(location: str) -> str:
    """
    Get mock weather information for a location.
    
    Args:
        location: The location to get weather for
        
    Returns:
        Mock weather information
    """
    # This is a mock tool for demonstration
    mock_weather = {
        "new york": "Sunny, 72°F (22°C), light breeze",
        "london": "Cloudy, 65°F (18°C), chance of rain",
        "tokyo": "Partly cloudy, 75°F (24°C), humid",
        "paris": "Overcast, 68°F (20°C), light wind"
    }
    
    location_lower = location.lower()
    if location_lower in mock_weather:
        return f"Weather in {location}: {mock_weather[location_lower]}"
    else:
        return f"Weather data not available for {location}. Available locations: New York, London, Tokyo, Paris"


def create_sample_agent():
    """Create a sample ReAct agent with basic tools."""
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    tools = [calculator, text_processor, weather_info]
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="You are a helpful assistant with access to calculator, text processing, and weather tools. Use the tools when appropriate to help answer user questions."
    )
    
    return agent


def example_successful_completion():
    """
    Example 1: Successful completion after review approval
    
    This example shows a straightforward case where the agent provides
    a good response that gets approved by the review agent.
    """
    print("=" * 60)
    print("EXAMPLE 1: Successful Completion After Review Approval")
    print("=" * 60)
    
    # Create the sample agent
    primary_agent = create_sample_agent()
    
    # Create two-stage workflow with default settings
    workflow = create_two_stage_workflow(
        primary_agent=primary_agent,
        max_retries=3
    )
    
    # Test with a clear mathematical question
    test_input = {
        "messages": [
            HumanMessage(content="What is 15 multiplied by 23? Please show the calculation.")
        ]
    }
    
    print("User Input: What is 15 multiplied by 23? Please show the calculation.")
    print("\nExecuting workflow...")
    print("-" * 40)
    
    # Execute the workflow
    result = workflow.invoke(test_input)
    
    # Display results
    print(f"Final Stage: {result.get('current_stage', 'unknown')}")
    print(f"Retry Count: {result.get('retry_count', 0)}")
    print(f"Review Result: {result.get('review_result', 'none')}")
    
    if result.get("messages"):
        print("\nFinal Response:")
        final_message = result["messages"][-1]
        if hasattr(final_message, 'content'):
            print(final_message.content)
        else:
            print(str(final_message))
    
    print("\n" + "=" * 60 + "\n")
    return result


def example_retry_loop():
    """
    Example 2: Retry loop when review fails
    
    This example demonstrates what happens when the initial response
    is not satisfactory and requires retries with feedback.
    """
    print("=" * 60)
    print("EXAMPLE 2: Retry Loop When Review Fails")
    print("=" * 60)
    
    # Create the sample agent
    primary_agent = create_sample_agent()
    
    # Create a strict review criteria that's more likely to reject responses
    strict_review_criteria = ReviewCriteria(
        max_retries=2,
        review_prompt="""
        Please review the following agent output with STRICT criteria:
        
        User Request: {user_request}
        Agent Output: {agent_output}
        
        The response must:
        1. Directly answer the question with specific details
        2. Show step-by-step work for calculations
        3. Provide context and explanation
        4. Be well-formatted and professional
        
        Be very strict in your evaluation. Only approve if ALL criteria are perfectly met.
        
        Respond with either:
        - "APPROVED: [brief reason]" if the output perfectly meets all criteria
        - "REJECTED: [specific detailed feedback]" if anything is missing or unclear
        """
    )
    
    # Create two-stage workflow with strict review
    workflow = create_two_stage_workflow(
        primary_agent=primary_agent,
        max_retries=2
    )
    workflow.review_criteria = strict_review_criteria
    
    # Test with a question that might need refinement
    test_input = {
        "messages": [
            HumanMessage(content="Calculate the area of a circle with radius 5 and explain the formula.")
        ]
    }
    
    print("User Input: Calculate the area of a circle with radius 5 and explain the formula.")
    print("\nExecuting workflow with strict review criteria...")
    print("-" * 40)
    
    # Execute the workflow and stream results to see the process
    for update in workflow.stream(test_input):
        for node_name, node_update in update.items():
            print(f"\n[{node_name.upper()}] Stage Update:")
            if "current_stage" in node_update:
                print(f"  Current Stage: {node_update['current_stage']}")
            if "retry_count" in node_update:
                print(f"  Retry Count: {node_update['retry_count']}")
            if "review_result" in node_update:
                print(f"  Review Result: {node_update['review_result']}")
            if "review_feedback" in node_update:
                print(f"  Review Feedback: {node_update['review_feedback'][:100]}...")
    
    print("\n" + "=" * 60 + "\n")


def example_custom_review_criteria():
    """
    Example 3: Custom review criteria and configuration
    
    This example shows how to customize the review criteria for
    specific use cases and requirements.
    """
    print("=" * 60)
    print("EXAMPLE 3: Custom Review Criteria and Configuration")
    print("=" * 60)
    
    # Create the sample agent
    primary_agent = create_sample_agent()
    
    # Create custom review criteria for weather-related queries
    weather_review_criteria = ReviewCriteria(
        max_retries=1,
        review_prompt="""
        Review this weather-related response:
        
        User Request: {user_request}
        Agent Output: {agent_output}
        
        For weather queries, the response should:
        1. Include temperature information
        2. Mention weather conditions (sunny, cloudy, etc.)
        3. Be helpful and informative
        4. Use the weather tool if asking about a specific location
        
        Respond with:
        - "APPROVED: [reason]" if it meets weather response criteria
        - "REJECTED: [specific feedback]" if it doesn't adequately address weather information
        """
    )
    
    # Create workflow with custom criteria
    workflow = create_two_stage_workflow(
        primary_agent=primary_agent,
        max_retries=1
    )
    workflow.review_criteria = weather_review_criteria
    
    # Test with weather query
    test_input = {
        "messages": [
            HumanMessage(content="What's the weather like in Tokyo today?")
        ]
    }
    
    print("User Input: What's the weather like in Tokyo today?")
    print("Using custom weather-focused review criteria...")
    print("-" * 40)
    
    # Execute the workflow
    result = workflow.invoke(test_input)
    
    # Display results
    print(f"Final Stage: {result.get('current_stage', 'unknown')}")
    print(f"Retry Count: {result.get('retry_count', 0)}")
    print(f"Review Result: {result.get('review_result', 'none')}")
    
    if result.get("review_feedback"):
        print(f"Review Feedback: {result['review_feedback']}")
    
    if result.get("messages"):
        print("\nConversation History:")
        for i, msg in enumerate(result["messages"]):
            role = getattr(msg, 'type', 'unknown') if hasattr(msg, 'type') else 'unknown'
            content = getattr(msg, 'content', str(msg)) if hasattr(msg, 'content') else str(msg)
            print(f"  {i+1}. [{role.upper()}]: {content[:100]}...")
    
    print("\n" + "=" * 60 + "\n")
    return result


def example_text_processing():
    """
    Example 4: Text processing with review
    
    This example demonstrates the workflow with text processing tasks.
    """
    print("=" * 60)
    print("EXAMPLE 4: Text Processing with Review")
    print("=" * 60)
    
    # Create the sample agent
    primary_agent = create_sample_agent()
    
    # Create workflow with moderate retry limit
    workflow = create_two_stage_workflow(
        primary_agent=primary_agent,
        max_retries=2
    )
    
    # Test with text processing request
    test_input = {
        "messages": [
            HumanMessage(content="Please convert 'Hello World' to uppercase and also count the words in it.")
        ]
    }
    
    print("User Input: Please convert 'Hello World' to uppercase and also count the words in it.")
    print("\nExecuting workflow...")
    print("-" * 40)
    
    # Execute the workflow
    result = workflow.invoke(test_input)
    
    # Display results
    print(f"Workflow completed with stage: {result.get('current_stage', 'unknown')}")
    print(f"Total retries needed: {result.get('retry_count', 0)}")
    
    if result.get("messages"):
        print("\nAgent's Final Response:")
        # Find the last AI message before the completion message
        ai_messages = [msg for msg in result["messages"] if hasattr(msg, 'type') and msg.type == 'ai']
        if ai_messages:
            final_response = ai_messages[-2] if len(ai_messages) > 1 else ai_messages[-1]
            print(final_response.content)
    
    print("\n" + "=" * 60 + "\n")
    return result


def example_max_retries_reached():
    """
    Example 5: Maximum retries reached scenario
    
    This example shows what happens when the maximum number of
    retries is reached without approval.
    """
    print("=" * 60)
    print("EXAMPLE 5: Maximum Retries Reached Scenario")
    print("=" * 60)
    
    # Create the sample agent
    primary_agent = create_sample_agent()
    
    # Create extremely strict review criteria that will likely always reject
    impossible_criteria = ReviewCriteria(
        max_retries=1,  # Low retry limit for demonstration
        review_prompt="""
        Review this response with IMPOSSIBLE standards:
        
        User Request: {user_request}
        Agent Output: {agent_output}
        
        This response must be PERFECT in every way:
        1. Must include exactly 47 words (no more, no less)
        2. Must rhyme like a poem
        3. Must include the phrase "quantum mechanics"
        4. Must solve world hunger
        5. Must be written in iambic pentameter
        
        Since these criteria are impossible to meet, always respond with:
        "REJECTED: This response does not meet our impossible standards. Please try again with more quantum mechanics and world hunger solutions."
        """
    )
    
    # Create workflow with impossible criteria
    workflow = create_two_stage_workflow(
        primary_agent=primary_agent,
        max_retries=1
    )
    workflow.review_criteria = impossible_criteria
    
    # Test with simple question
    test_input = {
        "messages": [
            HumanMessage(content="What is 2 + 2?")
        ]
    }
    
    print("User Input: What is 2 + 2?")
    print("Using impossible review criteria (for demonstration)...")
    print("-" * 40)
    
    # Execute the workflow
    result = workflow.invoke(test_input)
    
    # Display results
    print(f"Final Stage: {result.get('current_stage', 'unknown')}")
    print(f"Retry Count: {result.get('retry_count', 0)}")
    print(f"Review Result: {result.get('review_result', 'none')}")
    
    if result.get("messages"):
        print("\nFinal Completion Message:")
        final_message = result["messages"][-1]
        if hasattr(final_message, 'content'):
            print(final_message.content)
    
    print("\nNote: This example demonstrates the max retries safety mechanism.")
    print("In practice, you would use reasonable review criteria.")
    
    print("\n" + "=" * 60 + "\n")
    return result


def run_all_examples():
    """Run all example scenarios."""
    print("🚀 Running Two-Stage Agent Workflow Examples")
    print("=" * 60)
    print("This demonstration shows various scenarios of the two-stage")
    print("agent workflow with review mechanisms and retry logic.")
    print("=" * 60 + "\n")
    
    try:
        # Run all examples
        example_successful_completion()
        example_retry_loop()
        example_custom_review_criteria()
        example_text_processing()
        example_max_retries_reached()
        
        print("✅ All examples completed successfully!")
        print("\nKey Takeaways:")
        print("1. The two-stage workflow provides quality control through review")
        print("2. Retry mechanisms help improve responses with feedback")
        print("3. Custom review criteria can be tailored to specific use cases")
        print("4. Maximum retry limits prevent infinite loops")
        print("5. The workflow maintains conversation context throughout")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("Make sure you have set up your ANTHROPIC_API_KEY environment variable.")


if __name__ == "__main__":
    # Run all examples when script is executed directly
    run_all_examples()
