import os
from openai import OpenAI
from dotenv import load_dotenv

# loading the .env file
load_dotenv()

# initializing OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("🚀 Making an API call to OpenAI to check full responses...")

# calling the API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What's 25 + 37?"}
    ]
)

print("\n✅ Response from AI:")
print(response.choices[0].message.content)
print("\n-------------------")
print(response.model)
print("\n-------------------")
print(response.choices[0].finish_reason)
print("\n-------------------")
print(response.usage.total_tokens)
print("\n🎉 Success! The OpenAI connection works!")