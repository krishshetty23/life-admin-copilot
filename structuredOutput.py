import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# loading the .env file
load_dotenv()

# initializing OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


print("🚀 Making an API call to OpenAI to check structured responses...")
print("\n-------------------")

# calling the OpenAI API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {"role": "user", 
         "content": ("Give me information about Albert Einstein in JSON format with these fields: "
         "- name"
         "- birth_year"
         "- field"
         "- famous_for"
         "Return ONLY valid JSON, nothing else.")}
    ]
)

# converting the response into JSON
rawMessage = response.choices[0].message.content
data = json.loads(rawMessage)

# checking
print(data)
print("\n-------------------")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"}, 
    messages=[
        {
            "role": "user",
            "content": """
Give me the top 3 programming languages in JSON format:
{
  "languages": [
    {"name": "...", "year_created": ..., "use_case": "..."},
    ...
  ],
  "total_count": 3
}

Return ONLY valid JSON.
""",
        }
    ],
)

# converting the response into JSON
raw_message = response.choices[0].message.content
data = json.loads(raw_message)

print("\nParsed dict:")
print(data)
print("\n-------------------")
print(data["languages"][0]["name"])
print("\n-------------------")

print("Languages: ")
for lang in data["languages"]:
    print(lang["name"])
print("\n-------------------")

emailText = """Subject: Action Required: Address Update Needed

Dear Resident,

We need you to update your address in our system by March 15, 2024. 
There is a $25 processing fee. Please respond with your current address.

If you have questions, call us at 555-0123.

Thank you,
City Administration
"""

prompt = f"""
Extract the following information from this email and return as JSON:
{{
  "action_required": "...",
  "deadline": "YYY-MM-DD",
  "fee_amount": ...,
  "contact_phone": "..."
}}

Email:
{emailText}

Return ONLY valid JSON.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)

raw_message = response.choices[0].message.content
data = json.loads(raw_message)

print(data)
print("\n-------------------")