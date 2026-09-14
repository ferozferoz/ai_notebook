import ollama

# We will use the lightweight coding model you chose earlier
MODEL = 'qwen2.5-coder:1.5b'

print("Sending request to Ollama...")
response = ollama.chat(
    model=MODEL,
    messages=[{'role': 'user', 'content': 'Give me a 1-line Python code to reverse a list.'}]
)

# Extract and print the plain text answer
print("\n--- Model Response ---")
print(response['message']['content'])
