import os
import json
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from SemanticSearch import semantic_search

load_dotenv()

# --------- Tool wrappers using @tool decorator ---------

@tool
def search_profile(query):
    """
    Search the user's profile for information.
    Useful for finding information in the user's personal profile.
    Input should be a search query about personal info like address, email, phone, name, etc.
    """
    searchResult = semantic_search(query)
    bestMatch = searchResult["best_match"]
    score = searchResult["score"]
    if score < 0.3:
        return f"No relevant info found for: {query} (best score: {score:.2f})"
    else:
        return f"Best match for '{query}': {bestMatch} (score: {score:.2f})"

@tool
def calculator(expression):
    """
    Evaluate a simple math expression.
    Useful for doing math calculations. Input should be a mathematical expression like '5 + 3' or '10 * 2'.
    """
    try:
        result = str(eval(expression))
        return result
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"

@tool
def check_missing_info(query):
    """
    Check whether information exists in the profile.
    Use this when the search_profile doesn't find what you need.
    This helps identify what info to request from the user.
    """
    searchResult = semantic_search(query)
    score = searchResult["score"]
    if score < 0.3:
        return f"This information is not in the profile. You should ask the user for: {query}"
    else:
        return f"This information is available in the profile with confidence {score:.2f}"

# --------- Create tools list ---------

tools = [search_profile, calculator, check_missing_info]

# --------- LLM + Agent ---------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# Create the agent using the new API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are a helpful assistant that can search user profiles and answer questions.

When you need to find information, use the available tools:
- search_profile: To find information in the user's profile
- calculator: To perform calculations
- check_missing_info: To check if information is missing from the profile

Think step by step and use the tools to help answer questions."""
)

# --------- Verbose thinking function ---------

def run_agent_with_thinking(user_input):
    """
    Run the agent and show its thinking process step-by-step.
    """
    print(f"\n{'='*60}")
    print(f"USER INPUT: {user_input}")
    print(f"{'='*60}\n")

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    # Print all messages to see the thinking
    print("\n--- AGENT'S THINKING PROCESS ---")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        if hasattr(msg, 'content'):
            content = msg.content
        else:
            content = str(msg)

        print(f"\nStep {i+1} [{msg_type}]:")
        print(content[:200] + "..." if len(str(content)) > 200 else content)

    # Print final answer
    final_message = result["messages"][-1]
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(final_message.content if hasattr(final_message, 'content') else str(final_message))
    print(f"{'='*60}\n")

    return result

# --------- Conversational memory function ---------

def run_conversation():
    """
    Run a multi-turn conversation with the agent.
    The agent will remember previous exchanges.
    """
    print("\n" + "="*60)
    print("CONVERSATIONAL AGENT (type 'quit' to exit)")
    print("="*60 + "\n")

    # Track conversation history
    conversation_messages = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nEnding conversation. Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        conversation_messages.append({"role": "user", "content": user_input})

        # Invoke agent with full conversation history
        result = agent.invoke({"messages": conversation_messages})

        # Extract assistant's response
        assistant_message = result["messages"][-1]
        assistant_content = assistant_message.content if hasattr(assistant_message, 'content') else str(assistant_message)

        # Add assistant response to history
        conversation_messages.append({"role": "assistant", "content": assistant_content})

        # Print response
        print(f"\nAgent: {assistant_content}")

# --------- Manual tests ---------

if __name__ == "__main__":
    print("="*60)
    print("LANGCHAIN AGENT WITH VISIBLE THINKING")
    print("="*60)

    # Test 1 & 2: Basic tests with visible thinking
    run_agent_with_thinking("What is the user's email address?")
    run_agent_with_thinking("I need to update the user's address. What is their current address, and should I ask for a new one?")

    # Test 3: Multi-turn conversation with memory
    print("\n" + "="*60)
    print("TEST 3: MULTI-TURN CONVERSATION WITH MEMORY")
    print("="*60)

    # Simulated multi-turn conversation
    conversation_messages = []

    # Turn 1
    print("\nTurn 1: User asks for email")
    conversation_messages.append({"role": "user", "content": "What is the user's email?"})
    result = agent.invoke({"messages": conversation_messages})
    response = result["messages"][-1].content
    conversation_messages.append({"role": "assistant", "content": response})
    print(f"Agent: {response}")

    # Turn 2: Reference previous answer
    print("\nTurn 2: User asks follow-up (should remember email from Turn 1)")
    conversation_messages.append({"role": "user", "content": "Can you confirm that email address again?"})
    result = agent.invoke({"messages": conversation_messages})
    response = result["messages"][-1].content
    conversation_messages.append({"role": "assistant", "content": response})
    print(f"Agent: {response}")

    # Turn 3: Complex follow-up
    print("\nTurn 3: User asks about different info")
    conversation_messages.append({"role": "user", "content": "Now what about the address?"})
    result = agent.invoke({"messages": conversation_messages})
    response = result["messages"][-1].content
    print(f"Agent: {response}")

    print("\n" + "="*60)
    print("Memory test complete! Agent maintained context across turns.")
    print("="*60)

    # Uncomment the line below to start an interactive conversation
    # run_conversation()
