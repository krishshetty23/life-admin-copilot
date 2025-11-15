import os
import json
from typing import Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _validate_email_payload(data: Dict[str, Any]) -> None:
    """
    Validate that the parsed JSON has all required fields
    for the detected type. Raises ValueError if invalid.
    """
    if "type" not in data:
        raise ValueError("Missing 'type' field in response")

    email_type = data["type"]

    required_fields_by_type = {
        "address_change": ["deadline", "fee", "current_address", "contact"],
        "payment_confirmation": [
            "amount",
            "date",
            "invoice_number",
            "transaction_id",
            "payment_method",
        ],
        "appointment_reminder": [
            "date",
            "time",
            "location",
            "person",
            "purpose",
            "reschedule_notice",
            "contact",
        ],
    }

    if email_type not in required_fields_by_type:
        raise ValueError(f"Unknown email type: {email_type!r}")

    missing = [
        field
        for field in required_fields_by_type[email_type]
        if field not in data or data[field] in (None, "")
    ]

    if missing:
        raise ValueError(
            f"Missing required field(s) for type '{email_type}': {', '.join(missing)}"
        )


# function to parse email
def parseEmail(email_text: str) -> dict:
    """
    Parse an email into structured JSON.

    On success:
        returns the structured dict (one of the 3 formats).
    On failure:
        returns {
            "error": "Failed to parse email",
            "raw_response": "..."
        }
    """
    # We'll keep track of the raw model response for error reporting
    raw_response_text = ""

    try:
        prompt = f"""
You are an email parser.

Your job:
1. Decide which of these three types the email is:
   - "address_change"
   - "payment_confirmation"
   - "appointment_reminder"

2. Output EXACTLY ONE JSON object in the correct format:

If it is an address change email, output:
{{
  "type": "address_change",
  "deadline": "YYYY-MM-DD",
  "fee": 0.0,
  "current_address": "string",
  "contact": "string"
}}

If it is a payment confirmation email, output:
{{
  "type": "payment_confirmation",
  "amount": 0.0,
  "date": "YYYY-MM-DD",
  "invoice_number": "string",
  "transaction_id": "string",
  "payment_method": "string"
}}

If it is an appointment reminder email, output:
{{
  "type": "appointment_reminder",
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "location": "string",
  "person": "string",
  "purpose": "string",
  "reschedule_notice": "string",
  "contact": "string"
}}

Rules:
- Choose the correct "type" based on the email content.
- All dates MUST be in YYYY-MM-DD format.
- "time" MUST be in 24-hour HH:MM format (e.g., "14:30" for 2:30 PM).
- "fee" and "amount" MUST be numbers (no currency symbol).
- "reschedule_notice" should be like "48 hours" if present.
- "contact" should be the phone number used for contact/reschedule.
- Do NOT add any extra keys.
- Return ONLY valid JSON, nothing else.

Email:
{email_text}
"""

        # 1) API call — network/API error can happen here
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )

        # 2) Extract raw content from the model
        raw_response_text = response.choices[0].message.content or ""

        # 3) JSON parsing — model might still output something malformed
        try:
            data = json.loads(raw_response_text)
        except json.JSONDecodeError as e:
            # Parsing error
            raise ValueError(f"JSON parsing failed: {e}") from e

        # 4) Data validation — check required fields by type
        _validate_email_payload(data)

        # If we reached here, all good
        return data

    except Exception:
        # Bonus challenge: unified error shape
        return {
            "error": "Failed to parse email",
            "raw_response": raw_response_text,
        }


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
    print("Email 1 content: ")
    print("-------------------")
    print(parseEmail(email_1))
    print("-------------------")
    print("Email 2 content: ")
    print("-------------------")
    print(parseEmail(email_2))
    print("-------------------")
    print("Email 2 content: ")
    print("-------------------")
    print(parseEmail(email_3))
    print("-------------------")