import os
import json
from dotenv import load_dotenv

# Import all the building blocks
from EmailParser import parse_email
from AgenticLangChain import agent, search_profile, calculator, check_missing_info

load_dotenv()


def handle_email_end_to_end(raw_email):
    """
    Complete Life Admin Copilot pipeline:
    1. Parse email (Day 2)
    2. Agent analyzes and searches profile (Day 4)
    3. Generate intelligent reply

    Returns complete result with all metadata
    """
    print("\n" + "="*60)
    print("LIFE ADMIN COPILOT - END-TO-END PIPELINE")
    print("="*60 + "\n")

    # STEP 1: Parse the email
    print("STEP 1: Parsing email...")
    parsed = parse_email(raw_email)
    print(f"Email type detected: {parsed.get('email_type', 'unknown')}")
    print(f"Extracted fields: {list(parsed.keys())}")
    
    # STEP 2: Use agent to handle the email
    print("\nSTEP 2: Agent analyzing email and searching profile...")
    
    # Create a prompt for the agent based on parsed email
    agent_prompt = f"""
An email has been received with the following parsed information:

Email Type: {parsed.get('email_type', 'unknown')}
Content: {parsed.get('original_email', raw_email)[:200]}...

Extracted Information:
{json.dumps({k: v for k, v in parsed.items() if k not in ['original_email', 'parsing_status']}, indent=2)}

Your task:
1. Search the user's profile for relevant information
2. Determine what information is available vs missing
3. Generate a professional email reply that:
   - Acknowledges the email
   - Provides any requested information from the profile
   - Politely requests any missing information
   - Keeps it concise (2-3 sentences)
"""
    
    # Invoke agent
    result = agent.invoke({
        "messages": [{"role": "user", "content": agent_prompt}]
    })
    
    agent_response = result["messages"][-1].content
    
    print("\nSTEP 3: Generated reply:")
    print("-" * 60)
    print(agent_response)
    print("-" * 60)
    
    # Return complete result
    return {
        "parsed_email": parsed,
        "agent_response": agent_response,
        "full_result": result
    }


if __name__ == "__main__":
    # Test with different email types
    
    # Test 1: Address change request
    email1 = """
Subject: Address Update Required

Dear Customer,

We need to update your address in our system. Please confirm your current 
residential address at your earliest convenience.

Best regards,
Customer Service
"""
    
    result1 = handle_email_end_to_end(email1)
    
    print("\n\n" + "="*60)
    print("=" * 60 + "\n")
    
    # Test 2: Payment confirmation
    email2 = """
Subject: Payment Confirmation Needed

Hello,

We received a payment but need to verify the details. Could you please 
confirm your registered email address and phone number?

Thank you,
Billing Department
"""
    
    result2 = handle_email_end_to_end(email2)