import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("AI4Math/MathVista")["testmini"]

# Load Meta Llama 2 model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Adjust if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Store results in a list
results = {}

# Iterate through dataset one by one
for i, sample in enumerate(dataset):
    print(i)
    prompt = sample["query"]  # Assuming 'question' is the field to be used as input
    #print(prompt)
    # Generate response
    output = generator(prompt, max_length=1024, num_return_sequences=1)
    #print(output)
    # Store result
    #results.append({
     #   "example": i + 1,
      #  "question": prompt,
       # "generated_answer": output[0]['generated_text']
   # })
    results[str(i + 1)] = {
        "question": sample["question"],
        "image": sample.get("image", "none"),
        "choices": sample.get("choices", "none"),
        "unit": sample.get("unit", "none"),
        "precision": sample.get("precision", 1),
        "answer": sample.get("answer", "none"), #output[0]['generated_text'],
        "question_type": sample.get("question_type", "free_form"),
        "answer_type": sample.get("answer_type", "text"),
        "pid": sample.get("pid", str(i + 1)),
        "metadata": sample.get("metadata", {}),
        "query": sample.get("query", prompt),
        "response": output[0]['generated_text']
    }
    # Optional: Stop after a few examples
    #if i >= 5:  # Change this number to control how many examples to process
      # break

# Save results to a JSON file
    with open("/data/user_data/mjagadee/mmml/llama_answers.json", "w", encoding="utf-8") as f:
    	json.dump(results, f, indent=4, ensure_ascii=False)

print("Results saved to generated_answers.json")
