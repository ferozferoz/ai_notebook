import ollama

# We will use the lightweight coding model you chose earlier
MODEL = 'qwen2.5-coder:1.5b'

prompt_history = []

def call_ollama_chat(prompt):
    prompt_history.append({'role': 'user', 'content': prompt})
    response = ollama.chat(model=MODEL, messages=prompt_history)
    # save the response to a list prompt_history
    prompt_history.append({'role': 'user', 'content': response['message']['content']})

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        call_ollama_chat(user_input)
        print(prompt_history)
        print("Ollama:", prompt_history[-1]['content'])