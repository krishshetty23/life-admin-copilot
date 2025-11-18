from emailParser import ParseEmail
from semanticsearch import SemanticSearch
import os
from dotenv import load_dotenv
import boto3
import json


EMAIL_TYPE_TO_QUERY = {
    "address_change": "current address home location",
    "payment_confirmation": "payment bank account",
    "appointment_reminder": "schedule availability appointments",
}

SIMILARITY_THRESHOLD = 0.3  # minimum score to trust the retrieval

load_dotenv()
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('BEDROCK_IAM_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('BEDROCK_IAM_SECRET_ACCESS_KEY')
)


# function to process email with RAG
def ragProcessing(email_text):
    """
    Process an email with the RAG pipeline:
    - Parse the email into structured JSON
    - Build a semantic search query based on the email type

    Returns a dict like:
    {
        "parsed_email": {...},        # output from ParseEmail
        "search_query": "..." | None  # query string for semantic search
    }
    """
    # parsing the email
    parsedEmail = ParseEmail(email_text)

    # If parsing failed, propagate the error and don't build a query
    if isinstance(parsedEmail, dict) and "error" in parsedEmail:
        return {
            "parsed_email": parsedEmail,
            "search_query": None,
            "profile_result": None,
        }

    # building search query from the email type
    emailType = parsedEmail.get("type")
    searchQuery = EMAIL_TYPE_TO_QUERY.get(emailType, None)

    # semantic search over the profile
    if searchQuery:
        profileResult = SemanticSearch(searchQuery)

    return {
        "parsed_email": parsedEmail,
        "search_query": searchQuery,
        "profile_result": profileResult,
    }


# context matching
def evaluateContext(profile_result):
    """
    Evaluate whether retrieved context is relevant enough to use.
    
    Returns dict with:
    - is_relevant: bool
    - context: str or None
    - confidence: float
    """
    # If semantic search found nothing / wasn't run
    if profile_result is None:
        return {
            "is_relevant": False,
            "context": None,
            "confidence": 0.0,
        }
    
    similarityScore = float(profile_result.get("score", 0.0))
    bestMatch = profile_result.get("best_match")

    isRelevant = similarityScore >= SIMILARITY_THRESHOLD

    return {
        "is_relevant": isRelevant,
        "context": bestMatch if isRelevant else None,
        "confidence": similarityScore,
    }


# prompt generator
def genPrompt(email, eval):
    """
    Build a prompt for the LLM to generate an email reply.
    
    Combines:
    - Parsed email data
    - Retrieved context (if relevant)
    - Instructions for reply generation
    
    Returns: str (the full prompt)
    """
    # handling parser errors gracefully
    if email is None or "error" in email:
        emailType = "unknown"
        strInfo = "Could not reliably parse this email. Respond with a polite generic reply asking for clarification."
    else:
        emailType = email.get("type", "unknown")
        liensInfo: list[str] = []              # building key information section based on email type

        if emailType == "address_change":
            liensInfo.append(f"- Deadline: {email.get('deadline')}")
            liensInfo.append(f"- Fee: {email.get('fee')}")
            liensInfo.append(f"- Current Address: {email.get('current_address')}")
            liensInfo.append(f"- Contact: {email.get('contact')}")
        
        elif emailType == "payment_confirmation":
            liensInfo.append(f"- Transaction Amount: {email.get('amount')}")
            liensInfo.append(f"- Date of Transaction: {email.get('date')}")
            liensInfo.append(f"- Invoice Number: {email.get('invoice_number')}")
            liensInfo.append(f"- Transaction ID: {email.get('transaction_id')}")
            liensInfo.append(f"- Payment Method: {email.get('payment_method')}")

        elif emailType == "appointment_reminder":
            liensInfo.append(f"- Date: {email.get('date')}")
            liensInfo.append(f"- Time: {email.get('time')}")
            liensInfo.append(f"- Location: {email.get('location')}")
            liensInfo.append(f"- Person: {email.get('person')}")
            liensInfo.append(f"- Purpose: {email.get('purpose')}")
            liensInfo.append(f"- Reschedule notice: {email.get('reschedule_notice')}")
            liensInfo.append(f"- Contact phone: {email.get('contact')}")

        else:
            liensInfo.append("- No structured fields available. Write a generic but polite reply.")
        
        # filtering out null values
        liensInfo = [line for line in liensInfo if not line.endswith("None")]
        strInfo = "\n".join(liensInfo) if liensInfo else "No structured fields available."
    
    # available context
    isRelevant = eval.get('is_relevant', False) if eval else False
    if isRelevant:
        contextText = eval.get('context') or ""
        contextSection = f"User Profile Context: {contextText}"
    else:
        contextSection = "Note: No relevant user information found in profile."

    # final prompt
    prompt = (
        "You are an AI assistant helping the user manage and reply to emails.\n\n"
        "A. Email Details:\n"
        f"Email Type: {emailType}\n"
        "Key Information:\n"
        f"{strInfo}\n\n"
        "B. Context:\n"
        f"{contextSection}\n\n"
        "C. Instructions:\n"
        "Generate a polite, professional reply to this email.\n"
        "If context is available, use it to personalize the response.\n"
        "Keep the reply concise (2–3 sentences).\n"
        "Reply with only the email body text (no JSON, no extra commentary).\n"
    )

    return prompt


# generating the reply
def replyAI(prompt):
    """
    Call OpenAI to generate an email reply based on the prompt.
    
    Returns: str (the generated reply text)
    """
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
        replyTxt = response_body['content'][0]['text'] or ""
        return replyTxt.strip()
    
    except Exception as e:      # error handling
        print(f"[generate_reply] Error calling Bedrock: {e}")
        return "I'm sorry, but I'm currently unable to generate a reply. Please try again later."
    

# running the complete pipleine
def ragPipeline(email_text, email_name):
    """
    Demonstrate the complete RAG pipeline on a single email.
    Shows all intermediate steps and final result.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {email_name}")
    print('='*60)
    
    # Run the pipeline
    result = ragProcessing(email_text)
    contextEval = evaluateContext(result["profile_result"])
    promptTxt = genPrompt(result["parsed_email"], contextEval)
    replyTxt = replyAI(promptTxt)
    
    # Show results
    print(f"\n📧 Email Type: {result['parsed_email'].get('type', 'unknown')}")
    print(f"🔍 Search Query: {result['search_query']}")
    print(f"📊 Context Relevant: {contextEval['is_relevant']} (confidence: {contextEval['confidence']:.2f})")
    
    if contextEval['is_relevant']:
        print(f"💡 Retrieved Context: {contextEval['context']}")
    
    print(f"\n✉️  Generated Reply:")
    print("-" * 60)
    print(replyTxt)
    print("-" * 60)
    
    return replyTxt




# Example emails
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
    ragPipeline(email_1, "Address Change Request")
    ragPipeline(email_2, "Payment Confirmation")
    ragPipeline(email_3, "Appointment Reminder")