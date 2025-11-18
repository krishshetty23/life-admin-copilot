from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from semanticsearch import SemanticSearch
import os
from dotenv import load_dotenv
import boto3
import json

# Load environment variables
load_dotenv()

# Initialize Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('BEDROCK_IAM_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('BEDROCK_IAM_SECRET_ACCESS_KEY')
)

# Configuration
SIMILARITY_THRESHOLD = 0.3
MAX_SEARCH_ATTEMPTS = 3


# Define the ConversationState with search tracking
class ConversationState(TypedDict):
    current_query: str
    search_results: str
    search_score: float
    response: str
    search_attempts: int  # Track retry count


# Node 1: Parse Query
def parse_query_node(state: ConversationState):
    """
    Parse and prepare the user query for searching.
    Initialize search attempts counter.
    """
    current_query = state.get("current_query", "")

    print(f"\n[parse_query_node] Processing query: {current_query}")
    print(f"[parse_query_node] Initializing search attempts to 0")

    return {
        "search_attempts": 0
    }


# Node 2: Search Profile (with counter increment)
def search_profile_node(state: ConversationState):
    """
    Search the user profile for relevant information.
    Increments the search attempts counter each time.
    """
    current_query = state.get("current_query", "")
    attempts = state.get("search_attempts", 0) + 1  # Increment!

    print(f"\n[search_profile_node] Search attempt #{attempts}")
    print(f"[search_profile_node] Searching for: {current_query}")

    # Optionally refine query on retry
    search_query = current_query
    if attempts > 1:
        # On retry, try to broaden or refine the search
        print(f"[search_profile_node] This is a retry - using refined query")
        search_query = current_query  # Could modify this to be smarter

    # Run semantic search
    try:
        search_result = SemanticSearch(search_query)

        if search_result and search_result.get("score", 0) >= SIMILARITY_THRESHOLD:
            best_match = search_result.get("best_match", "")
            score = search_result.get("score", 0)
            search_results = f"Found relevant information (confidence: {score:.2f}): {best_match}"
            print(f"[search_profile_node] Success! Score: {score:.2f}")
        else:
            search_results = "No relevant information found in profile."
            score = search_result.get("score", 0) if search_result else 0.0
            print(f"[search_profile_node] No good results. Score: {score:.2f}")

    except Exception as e:
        print(f"[search_profile_node] Search error: {e}")
        search_results = "Error searching profile."
        score = 0.0

    return {
        "search_results": search_results,
        "search_score": score,
        "search_attempts": attempts
    }


# Node 3: Evaluate Results
def evaluate_results_node(state: ConversationState):
    """
    Evaluate the quality of search results.
    This node doesn't modify state, just logs the evaluation.
    The conditional edge function will make the routing decision.
    """
    search_results = state.get("search_results", "")
    score = state.get("search_score", 0.0)
    attempts = state.get("search_attempts", 0)

    has_good_results = "Found relevant information" in search_results

    print(f"\n[evaluate_results_node] Evaluating search results")
    print(f"[evaluate_results_node] Attempts so far: {attempts}/{MAX_SEARCH_ATTEMPTS}")
    print(f"[evaluate_results_node] Score: {score:.2f}")
    print(f"[evaluate_results_node] Has good results: {has_good_results}")

    return {}


# CONDITIONAL ROUTING FUNCTION - THE MAGIC!
def should_continue_searching(state: ConversationState) -> Literal["search_profile", "generate_response"]:
    """
    Decides whether to loop back to search again or continue to generate response.

    Returns the NAME of the next node to execute.
    """
    search_results = state.get("search_results", "")
    attempts = state.get("search_attempts", 0)
    score = state.get("search_score", 0.0)

    # Check if results are good enough
    has_good_results = "Found relevant information" in search_results

    print(f"\n[ROUTING DECISION]")
    print(f"  - Attempts: {attempts}/{MAX_SEARCH_ATTEMPTS}")
    print(f"  - Score: {score:.2f}")
    print(f"  - Has good results: {has_good_results}")

    # Decision logic
    if attempts >= MAX_SEARCH_ATTEMPTS:
        print(f"  - DECISION: Max attempts reached -> generate_response")
        return "generate_response"  # Give up, generate with what we have

    if has_good_results:
        print(f"  - DECISION: Good results found -> generate_response")
        return "generate_response"  # Success! Move forward
    else:
        print(f"  - DECISION: Poor results, trying again -> search_profile")
        return "search_profile"  # Loop back and try again


# Node 4: Generate Response
def generate_response_node(state: ConversationState):
    """
    Generate a response using Bedrock based on search results.
    If no good results were found after max attempts, admits it doesn't know.
    """
    current_query = state.get("current_query", "")
    search_results = state.get("search_results", "")
    attempts = state.get("search_attempts", 0)

    print(f"\n[generate_response_node] Generating response")
    print(f"[generate_response_node] Used {attempts} search attempt(s)")

    # Check if we have good results
    has_good_results = "Found relevant information" in search_results

    # Build prompt
    if has_good_results:
        prompt = f"""You are a helpful AI assistant managing a user's personal information.

User Query: {current_query}

Retrieved Information: {search_results}

Generate a helpful, concise response (2-3 sentences) using the retrieved information."""
    else:
        prompt = f"""You are a helpful AI assistant managing a user's personal information.

User Query: {current_query}

Note: After {attempts} search attempts, no relevant information was found in the user's profile.

Generate a polite response admitting you don't have this information and suggesting what the user could do."""

    # Call Bedrock
    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
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

        return {"response": response_text}

    except Exception as e:
        print(f"[generate_response_node] Error calling Bedrock: {e}")
        return {"response": "I apologize, but I'm having trouble generating a response right now."}


# Node 5: Format Output
def format_output_node(state: ConversationState):
    """
    Display final results.
    """
    response = state.get("response", "")
    attempts = state.get("search_attempts", 0)
    score = state.get("search_score", 0.0)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Search Attempts: {attempts}")
    print(f"Final Score: {score:.2f}")
    print(f"\nResponse:")
    print(response)
    print("=" * 60)

    return {}


# Build the graph with cycles
def create_cyclical_graph():
    """
    Create a graph with conditional routing and cycles.
    """
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("search_profile", search_profile_node)
    workflow.add_node("evaluate_results", evaluate_results_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("format_output", format_output_node)

    # Set entry point
    workflow.set_entry_point("parse_query")

    # Add regular edges
    workflow.add_edge("parse_query", "search_profile")
    workflow.add_edge("search_profile", "evaluate_results")

    # Add CONDITIONAL edge - this is where the magic happens!
    workflow.add_conditional_edges(
        "evaluate_results",  # From this node
        should_continue_searching,  # Use this function to decide
        {
            "search_profile": "search_profile",  # If function returns "search_profile", loop back
            "generate_response": "generate_response"  # If function returns "generate_response", move forward
        }
    )

    # Continue to output
    workflow.add_edge("generate_response", "format_output")
    workflow.add_edge("format_output", END)

    # Compile (no checkpointer needed for this demo)
    app = workflow.compile()

    return app


# Test the cyclical graph
def test_cycles():
    """
    Test the graph with queries that will and won't find results.
    """
    print("\n" + "=" * 60)
    print("TESTING LANGGRAPH WITH CYCLES & CONDITIONAL ROUTING")
    print("=" * 60)

    app = create_cyclical_graph()

    # Test 1: Query that SHOULD find results
    print("\n\n" + "=" * 60)
    print("TEST 1: Query that should succeed")
    print("=" * 60)
    result_1 = app.invoke({
        "current_query": "What's my current address?",
        "search_results": "",
        "search_score": 0.0,
        "response": "",
        "search_attempts": 0
    })

    # Test 2: Query that WON'T find results (should loop 3 times then give up)
    print("\n\n" + "=" * 60)
    print("TEST 2: Query that should fail and retry 3 times")
    print("=" * 60)
    result_2 = app.invoke({
        "current_query": "What's my favorite ice cream flavor?",
        "search_results": "",
        "search_score": 0.0,
        "response": "",
        "search_attempts": 0
    })

    # Test 3: Query about phone number
    print("\n\n" + "=" * 60)
    print("TEST 3: Query about phone number")
    print("=" * 60)
    result_3 = app.invoke({
        "current_query": "Do you have my phone number?",
        "search_results": "",
        "search_score": 0.0,
        "response": "",
        "search_attempts": 0
    })

    print("\n\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\nKey observations:")
    print("- Successful queries should find results on attempt 1")
    print("- Failed queries should retry 3 times before giving up")
    print("- The graph intelligently routes based on result quality")


if __name__ == "__main__":
    test_cycles()
