import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv(override=True)
api_key = os.getenv('OLLAMA_API_KEY')


client = Client()
messages = [
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
    print(part["message"]["content"], end="", flush=True)
