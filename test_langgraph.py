from typing import TypedDict
from langgraph.graph import StateGraph, END
from emailParser import ParseEmail

# Email type to search query mapping (from your RAG pipeline)
EMAIL_TYPE_TO_QUERY = {
    "address_change": "current address home location",
    "payment_confirmation": "payment bank account",
    "appointment_reminder": "schedule availability appointments",
}


# 1. Define your state
class EmailState(TypedDict):
    email_text: str
    email_type: str
    search_query: str


# 2. Define your nodes (functions that modify state)
def parse_email_node(state: EmailState):
    """
    Node 1: Parse the email and extract the email type
    """
    email_text = state["email_text"]

    # Use your existing ParseEmail function
    parsed_email = ParseEmail(email_text)

    # Handle parsing errors
    if isinstance(parsed_email, dict) and "error" in parsed_email:
        print(f"[parse_email_node] Error: {parsed_email['error']}")
        return {"email_type": "unknown"}

    # Extract the type
    email_type = parsed_email.get("type", "unknown")
    print(f"[parse_email_node] Detected email type: {email_type}")

    return {"email_type": email_type}


def build_query_node(state: EmailState):
    """
    Node 2: Build search query based on email type
    """
    email_type = state["email_type"]

    # Map email type to search query
    search_query = EMAIL_TYPE_TO_QUERY.get(email_type, None)

    if search_query:
        print(f"[build_query_node] Search query for '{email_type}': {search_query}")
    else:
        print(f"[build_query_node] No search query defined for type: {email_type}")
        search_query = ""

    return {"search_query": search_query}


def print_result_node(state: EmailState):
    """
    Node 3: Print the final search query
    """
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Email Type: {state['email_type']}")
    print(f"Search Query: {state['search_query']}")
    print("=" * 60)

    return {}


# 3. Build the graph
workflow = StateGraph(EmailState)

# Add nodes
workflow.add_node("parse", parse_email_node)
workflow.add_node("build_query", build_query_node)
workflow.add_node("print", print_result_node)

# Connect nodes (edges)
workflow.set_entry_point("parse")
workflow.add_edge("parse", "build_query")
workflow.add_edge("build_query", "print")
workflow.add_edge("print", END)

# Compile the graph
app = workflow.compile()


# Test emails (from your existing examples)
email_1 = """Subject: Action Required - Update Your Address

Dear Resident,

Our records show your address may be outdated. Please update your
mailing address by March 20, 2024. There is a $15 processing fee.

Current address on file: 123 Oak St, Unit 5B, Chicago, IL 60601

To update, reply with your new address or call 1-800-555-0199.

Regards,
County Records Office
"""

email_2 = """Subject: Payment Receipt #INV-2024-0891

Hello,

This confirms your payment of $450.00 received on February 28, 2024
for Service Request #SR-4421.

Payment method: Credit Card ending in 4532
Transaction ID: TXN-88294-2024

Thank you for your payment.

Billing Department
"""

email_3 = """Subject: Reminder: Your appointment is scheduled

Hi there,

This is a reminder that your appointment is scheduled for:

Date: April 5, 2024
Time: 2:30 PM
Location: City Hall, Room 301
With: Inspector J. Martinez
Regarding: Building Permit Review

Please bring your permit application documents. If you need to
reschedule, call us at least 48 hours in advance at (555) 123-4567.

See you soon!
"""


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING LANGGRAPH WORKFLOW")
    print("=" * 60)

    # Test 1: Address Change Email
    print("\n\nTEST 1: Address Change Email")
    print("-" * 60)
    result_1 = app.invoke({
        "email_text": email_1,
        "email_type": "",
        "search_query": ""
    })

    # Test 2: Payment Confirmation Email
    print("\n\nTEST 2: Payment Confirmation Email")
    print("-" * 60)
    result_2 = app.invoke({
        "email_text": email_2,
        "email_type": "",
        "search_query": ""
    })

    # Test 3: Appointment Reminder Email
    print("\n\nTEST 3: Appointment Reminder Email")
    print("-" * 60)
    result_3 = app.invoke({
        "email_text": email_3,
        "email_type": "",
        "search_query": ""
    })

    print("\n\nAll tests completed!")
