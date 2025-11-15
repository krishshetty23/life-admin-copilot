import os
from openai import OpenAI
from dotenv import load_dotenv

# loading the API key from the .env
load_dotenv()

# initializing OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("🚀 Making my first API call to OpenAI...")

# first API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say 'Hello! I'm connected and ready to build!' in an excited way"}
    ]
)

print("\n✅ Response from AI:")
print(response.choices[0].message.content)
print("\n🎉 Success! The OpenAI connection works!")