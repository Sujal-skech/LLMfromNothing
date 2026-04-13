from transformers import pipeline

# Load model
pipe = pipeline(
    "text-generation",
    model="./tinyLLM",
    tokenizer="./tinyLLM",
)

print("🤖 Chat started (type 'exit' to quit)\n")

history = ""

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("👋 Chat ended")
        break

    # Add user input to history
    history += f"User: {user}\nAssistant:"

    # Generate response
    out = pipe(
        history,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    # Extract only new response
    response = out[0]["generated_text"]
    answer = response[len(history):]

    print("Bot:", answer.strip())

    # Save response in history
    history += answer + "\n"