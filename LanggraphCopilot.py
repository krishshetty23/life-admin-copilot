from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from SemanticSearch import semantic_search
import os
from dotenv import load_dotenv
import boto3
import json
import operator

# Load environment variables
load_dotenv()

# Initialize Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('BEDROCK_IAM_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('BEDROCK_IAM_SECRET_ACCESS_KEY')
)

SIMILARITY_THRESHOLD = 0.3


# Define the ConversationState with message history
class ConversationState(TypedDict):
    # Use Annotated with operator.add to append to list instead of replacing
    messages: Annotated[list[dict], operator.add]
    current_query: str
    search_results: str
    response: str


# Helper function to format message history
def format_messages(messages):
    """Format message history for display"""
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role.upper()}: {content}")
    return "\n".join(formatted)


# Node 1: Process Query
def process_query_node(state):
    """
    Process the incoming user query and add it to message history.
    Determine if semantic search is needed based on query keywords.
    """
    current_query = state.get("current_query", "")

    print(f"\n[process_query_node] Processing query: {current_query}")

    # Add user query to message history
    new_message = {"role": "user", "content": current_query}

    # Determine if we need semantic search (keywords that suggest profile lookup)
    search_keywords = ["address", "phone", "appointment", "payment", "bank", "schedule", "location", "contact"]
    needs_search = any(keyword in current_query.lower() for keyword in search_keywords)

    if needs_search:
        print(f"[process_query_node] Query requires profile search")
    else:
        print(f"[process_query_node] Query doesn't require profile search")

    return {
        "messages": [new_message],
        "search_results": "" if not needs_search else state.get("search_results", "")
    }


# Node 2: Search Profile
def search_profile_node(state):
    """
    Run semantic search on the user's query to find relevant profile information.
    """
    current_query = state.get("current_query", "")

    print(f"[search_profile_node] Searching profile for: {current_query}")

    # Run semantic search
    try:
        search_result = semantic_search(current_query)

        if search_result and search_result.get("score", 0) >= SIMILARITY_THRESHOLD:
            best_match = search_result.get("best_match", "")
            score = search_result.get("score", 0)
            search_results = f"Found relevant information (confidence: {score:.2f}): {best_match}"
            print(f"[search_profile_node] Found: {best_match} (score: {score:.2f})")
        else:
            search_results = "No relevant information found in profile."
            print(f"[search_profile_node] No relevant results")
    except Exception as e:
        print(f"[search_profile_node] Search error: {e}")
        search_results = "Error searching profile."

    return {"search_results": search_results}


# Node 3: Generate Response
def generate_response_node(state):
    """
    Generate a response using Bedrock, incorporating:
    - Full conversation history
    - Retrieved profile information
    - Context awareness
    """
    messages = state.get("messages", [])
    current_query = state.get("current_query", "")
    search_results = state.get("search_results", "")

    print(f"[generate_response_node] Generating response with {len(messages)} messages in history")

    # Build context-aware prompt
    conversation_history = format_messages(messages) if messages else "No previous conversation"

    prompt = f"""You are a helpful AI assistant managing a user's personal information and emails.

Previous conversation:
{conversation_history}

Retrieved profile information:
{search_results}

Instructions:
- Answer the user's latest query naturally and conversationally
- If you found relevant profile information, use it in your response
- If this is a follow-up question, reference previous context when appropriate
- Keep responses concise (2-4 sentences)
- Be friendly and professional

Generate your response:"""

    # Call Bedrock
    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = bedrock.invoke_model(
            modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=json.dumps(request_body)
        )

        response_body = json.loads(response['body'].read())
        response_text = response_body['content'][0]['text']

        print(f"[generate_response_node] Response generated successfully")

        # Add assistant response to message history
        assistant_message = {"role": "assistant", "content": response_text}

        return {
            "messages": [assistant_message],
            "response": response_text
        }

    except Exception as e:
        print(f"[generate_response_node] Error calling Bedrock: {e}")
        error_response = "I apologize, but I'm having trouble generating a response right now."
        return {
            "messages": [{"role": "assistant", "content": error_response}],
            "response": error_response
        }


# Node 4: Format Output
def format_output_node(state):
    """
    Format the final output for the user.
    """
    response = state.get("response", "")
    messages = state.get("messages", [])

    print("\n" + "=" * 60)
    print("CONVERSATION TURN COMPLETE")
    print("=" * 60)
    print(f"Total messages in history: {len(messages)}")
    print(f"\nAssistant: {response}")
    print("=" * 60)

    return {}


# Build the graph
def create_copilot_graph():
    """
    Create and compile the conversational copilot graph with memory.
    """
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("process_query", process_query_node)
    workflow.add_node("search_profile", search_profile_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("format_output", format_output_node)

    # Connect nodes (linear flow for now)
    workflow.set_entry_point("process_query")
    workflow.add_edge("process_query", "search_profile")
    workflow.add_edge("search_profile", "generate_response")
    workflow.add_edge("generate_response", "format_output")
    workflow.add_edge("format_output", END)

    # Add memory checkpointer
    memory = MemorySaver()

    # Compile with checkpointer
    app = workflow.compile(checkpointer=memory)

    return app


# Test the conversational copilot
def test_conversation():
    """
    Test multi-turn conversation with memory.
    """
    print("\n" + "=" * 60)
    print("TESTING CONVERSATIONAL COPILOT WITH MEMORY")
    print("=" * 60)

    # Create the copilot
    app = create_copilot_graph()

    # Session configuration (same thread_id = shared memory)
    config = {"configurable": {"thread_id": "user_session_1"}}

    # Turn 1: Ask about address
    print("\n\n>>> TURN 1: User asks about address")
    print("-" * 60)
    result_1 = app.invoke({
        "messages": [],
        "current_query": "What's my current address?",
        "search_results": "",
        "response": ""
    }, config)

    # Turn 2: Ask about phone number (should remember context)
    print("\n\n>>> TURN 2: User asks about phone number")
    print("-" * 60)
    result_2 = app.invoke({
        "current_query": "Do you have my phone number?",
        "search_results": "",
        "response": ""
    }, config)

    # Turn 3: Ask for summary (should use full history)
    print("\n\n>>> TURN 3: User asks for summary")
    print("-" * 60)
    result_3 = app.invoke({
        "current_query": "Summarize what you know about me",
        "search_results": "",
        "response": ""
    }, config)

    # Turn 4: Follow-up question
    print("\n\n>>> TURN 4: Follow-up question")
    print("-" * 60)
    result_4 = app.invoke({
        "current_query": "When is my next appointment?",
        "search_results": "",
        "response": ""
    }, config)

    print("\n\n" + "=" * 60)
    print("CONVERSATION COMPLETE")
    print("=" * 60)
    print(f"\nFinal message count: {len(result_4.get('messages', []))}")
    print("\nFull conversation history:")
    print("-" * 60)
    print(format_messages(result_4.get('messages', [])))
    print("-" * 60)


if __name__ == "__main__":
    test_conversation()
