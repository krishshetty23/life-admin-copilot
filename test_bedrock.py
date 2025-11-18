import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

# creating the Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('BEDROCK_IAM_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('BEDROCK_IAM_SECRET_ACCESS_KEY')
)

# test prompt
prompt = "Explain what AWS Bedrock is in one sentence."

# specific msg format for Bedrock
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

# calling Bedrock
response = bedrock.invoke_model(
    modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
    body=json.dumps(request_body)
)

# parsing the response
response_body = json.loads(response['body'].read())

# Extract the actual text
claude_response = response_body['content'][0]['text']
print("\n=== CLAUDE'S RESPONSE ===")
print(claude_response)
print(f"\n=== TOKENS USED: {response_body['usage']['input_tokens']} in, {response_body['usage']['output_tokens']} out ===")