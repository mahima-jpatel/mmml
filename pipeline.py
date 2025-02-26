from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("AI4Math/MathVista")["testmini"]

# Load Meta Llama 2 model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Adjust if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Iterate through dataset one by one
for i, sample in enumerate(dataset):
    prompt = sample["question"]  # Assuming 'question' is the field to be used as input

    # Generate response
    output = generator(prompt, max_length=1024, num_return_sequences=1)

    # Print output
    print(f"Example {i + 1}:")
    print(f"Question: {prompt}")
    print(f"Generated Answer: {output[0]['generated_text']}\n")

    # Optional: Stop after a few examples
    if i >= 5:  # Change this number to control how many examples to process
        break
