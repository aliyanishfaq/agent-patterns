#!/usr/bin/env python3
"""
Example script demonstrating the TwoStageReviewAgent workflow.

This script shows how to use the generic two-stage agent workflow with different
ReAct agents and tools, demonstrating the complete process from initial agent
execution through review and potential iteration.
"""

import os
from typing import List
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# Import our TwoStageReviewAgent
from agent import TwoStageReviewAgent


# Example tools for demonstration
@tool
def weather_lookup(location: str) -> str:
    """
    Get weather information for a location.
    
    Args:
        location: The city or location to get weather for
    
    Returns:
        Weather information for the location
    """
    # Simulated weather data
    weather_data = {
        "new york": "Sunny, 72°F (22°C), light breeze",
        "london": "Cloudy, 59°F (15°C), chance of rain",
        "tokyo": "Partly cloudy, 68°F (20°C), humid",
        "paris": "Overcast, 61°F (16°C), light rain",
        "sydney": "Clear, 75°F (24°C), windy"
    }
    
    location_lower = location.lower()
    for city, weather in weather_data.items():
        if city in location_lower:
            return f"Weather in {location}: {weather}"
    
    return f"Weather data not available for {location}. Try: New York, London, Tokyo, Paris, or Sydney."


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between different units of measurement.
    
    Args:
        value: The numeric value to convert
        from_unit: The unit to convert from (celsius, fahrenheit, meters, feet, kg, lbs)
        to_unit: The unit to convert to
    
    Returns:
        The converted value with units
    """
    conversions = {
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("meters", "feet"): lambda x: x * 3.28084,
        ("feet", "meters"): lambda x: x / 3.28084,
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x / 2.20462,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    else:
        available = ", ".join([f"{f} to {t}" for f, t in conversions.keys()])
        return f"Conversion not supported. Available conversions: {available}"


@tool
def simple_search(query: str) -> str:
    """
    Perform a simple search and return relevant information.
    
    Args:
        query: The search query
    
    Returns:
        Search results or information
    """
    # Simulated search results
    search_data = {
        "python": "Python is a high-level programming language known for its simplicity and readability. Created by Guido van Rossum in 1991.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It's designed for creating agent and multi-agent workflows.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines programmed to think and learn.",
        "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
        "react": "ReAct (Reasoning and Acting) is a paradigm that combines reasoning and acting in language models for better problem-solving."
    }
    
    query_lower = query.lower()
    for topic, info in search_data.items():
        if topic in query_lower:
            return f"Search result for '{query}': {info}"
    
    return f"No specific information found for '{query}'. Try searching for: Python, LangGraph, AI, Machine Learning, or ReAct."


def demonstrate_basic_workflow():
    """Demonstrate the basic two-stage workflow with simple tools."""
    print("=" * 60)
    print("BASIC TWO-STAGE WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    # Create a two-stage agent with basic tools
    agent = TwoStageReviewAgent(
        agent_creator=create_react_agent,
        agent_tools=[weather_lookup, unit_converter, simple_search],
        model_name="claude-3-5-sonnet-20241022",
        max_iterations=2
    )
    
    # Test queries that should pass review
    good_queries = [
        "What's the weather like in New York?",
        "Convert 25 celsius to fahrenheit",
        "Tell me about Python programming language"
    ]
    
    print("\n🟢 Testing queries that should PASS review:")
    for i, query in enumerate(good_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        try:
            result = agent.invoke({"messages": [HumanMessage(content=query)]})
            print(f"✅ Final Output: {result['initial_output'][:100]}...")
            print(f"📝 Review: {result['review_result'][:80]}...")
            print(f"🔄 Iterations: {result['iteration_count']}")
        except Exception as e:
            print(f"❌ Error: {e}")


def demonstrate_iteration_workflow():
    """Demonstrate the workflow with queries that might require iteration."""
    print("\n" + "=" * 60)
    print("ITERATION WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    # Create a two-stage agent with stricter review (by using fewer tools)
    agent = TwoStageReviewAgent(
        agent_creator=create_react_agent,
        agent_tools=[weather_lookup],  # Limited tools to potentially trigger iterations
        model_name="claude-3-5-sonnet-20241022",
        max_iterations=3
    )
    
    # Test queries that might require iteration
    challenging_queries = [
        "What's the weather in London and also convert 20 celsius to fahrenheit?",
        "Give me weather for Tokyo and tell me about machine learning"
    ]
    
    print("\n🟡 Testing queries that might require ITERATION:")
    for i, query in enumerate(challenging_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        try:
            result = agent.invoke({"messages": [HumanMessage(content=query)]})
            print(f"📊 Final Output: {result['initial_output'][:100]}...")
            print(f"📝 Review Result: {result['review_result'][:80]}...")
            print(f"✅ Review Passed: {result['review_passed']}")
            print(f"🔄 Total Iterations: {result['iteration_count']}")
            
            if result['iteration_count'] > 0:
                print("🔁 This query required iteration!")
            else:
                print("✨ Passed on first attempt!")
                
        except Exception as e:
            print(f"❌ Error: {e}")


def demonstrate_custom_agent():
    """Demonstrate using a custom agent creator function."""
    print("\n" + "=" * 60)
    print("CUSTOM AGENT DEMONSTRATION")
    print("=" * 60)
    
    def custom_agent_creator(model, tools):
        """Custom agent creator with specific configuration."""
        return create_react_agent(
            model=model,
            tools=tools,
            # Could add custom system message or other configurations here
        )
    
    # Create agent with custom creator
    agent = TwoStageReviewAgent(
        agent_creator=custom_agent_creator,
        agent_tools=[weather_lookup, unit_converter, simple_search],
        model_name="claude-3-5-sonnet-20241022",
        max_iterations=2
    )
    
    print("\n🔧 Testing with custom agent creator:")
    query = "Search for information about ReAct and convert 100 fahrenheit to celsius"
    print(f"Query: {query}")
    
    try:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"📊 Final Output: {result['initial_output'][:150]}...")
        print(f"📝 Review: {result['review_result'][:100]}...")
        print(f"🔄 Iterations: {result['iteration_count']}")
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_workflow_state():
    """Demonstrate accessing and understanding the workflow state."""
    print("\n" + "=" * 60)
    print("WORKFLOW STATE DEMONSTRATION")
    print("=" * 60)
    
    agent = TwoStageReviewAgent(
        agent_creator=create_react_agent,
        agent_tools=[weather_lookup, simple_search],
        model_name="claude-3-5-sonnet-20241022",
        max_iterations=2
    )
    
    query = "What's the weather in Paris and search for AI information?"
    print(f"Query: {query}")
    
    try:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        
        print("\n📋 Complete Workflow State:")
        print(f"🔤 Original Query: {result['original_query']}")
        print(f"📤 Initial Output: {result['initial_output'][:100]}...")
        print(f"📝 Review Result: {result['review_result'][:100]}...")
        print(f"✅ Review Passed: {result['review_passed']}")
        print(f"🔄 Iteration Count: {result['iteration_count']}")
        print(f"🎯 Max Iterations: {result['max_iterations']}")
        print(f"💬 Message Count: {len(result['messages'])}")
        
        # Show message types
        message_types = [type(msg).__name__ for msg in result['messages']]
        print(f"📨 Message Types: {message_types}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function to run all demonstrations."""
    print("🚀 Two-Stage Review Agent - Example Demonstrations")
    print("This script shows various ways to use the TwoStageReviewAgent")
    
    # Check if we have the required environment variable
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please set your Anthropic API key to run this example:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nContinuing with demonstration (may fail without API key)...")
    
    try:
        # Run all demonstrations
        demonstrate_basic_workflow()
        demonstrate_iteration_workflow()
        demonstrate_custom_agent()
        demonstrate_workflow_state()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED")
        print("=" * 60)
        print("\n📚 Key Takeaways:")
        print("1. The TwoStageReviewAgent accepts any ReAct agent creator function")
        print("2. It automatically handles review and iteration logic")
        print("3. You can customize tools, models, and iteration limits")
        print("4. The workflow state provides complete visibility into the process")
        print("5. Failed reviews trigger automatic retry with feedback")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        print("This might be due to missing API keys or network issues.")


if __name__ == "__main__":
    main()
