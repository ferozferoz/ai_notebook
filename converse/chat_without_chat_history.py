import ollama

# We will use the lightweight coding model you chose earlier
MODEL = 'qwen2.5-coder:1.5b'




if __name__=="__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = ollama.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': user_input}]
        )

        print("Ollama:", response['message']['content'])