import torch
import json
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==== STEP 1: Load Dataset ====
print("🔍 Loading MathVista dataset...")
dataset = load_dataset("AI4Math/MathVista")["testmini"]

# Convert dataset to a list of dictionaries
dataset = list(dataset)  # ✅ FIX HERE

print(f"✅ Loaded {len(dataset)} questions from MathVista.")

# ==== STEP 2: Load Model ====
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
print(f"🔄 Loading Llama-2-7B model: {MODEL_NAME}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Fix: Set padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding

# Load model (Choose CPU if debugging CUDA issues)
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    ).eval()
print("✅ Model loaded successfully (using GPU).")
print(model)

# ==== STEP 3: Run Inference ====
print("🚀 Starting inference...")

batch_size = 1  # ✅ Reduce batch size for debugging
batch_questions = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]

responses = []

for batch in tqdm(batch_questions, desc="🧠 Processing MathVista"):
    print(f"DEBUG: Batch sample: {batch[:1]}")  # Print first item in batch for debugging
    start_time = time.time()

    try:
        batch_inputs = [q["query"] for q in batch]  # Ensure each item has a 'query' ke
        print("batch_inputs", batch_inputs)
# ✅ Fix: Ensure padding token is used and add attention mask
        inputs = tokenizer(
            batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        #print("input", inputs.input_ids)
        input_ids = inputs.input_ids.cuda() 
        #print("input_ids", inputs.input_ids)
        attention_mask = inputs.attention_mask.cuda()
        #print("input_ids", input_ids)
        #print("attention_mask", attention_mask)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, max_new_tokens=5000, temperature=0.7,  # Adjust creativity of responses
    top_p=0.9,  # Nucleus sampling to focus on high-probability words
    do_sample=True)

        batch_responses = tokenizer.batch_decode(outputs)
        print(batch_responses, "Batch Resp")
        for i, q in enumerate(batch):
            responses.append({"id": q["pid"], "question": q["query"], "response": batch_responses[i]})

        elapsed_time = time.time() - start_time
        print(f"✅ Processed batch of {batch_size} in {elapsed_time:.2f} seconds.")
        break
    except Exception as e:
        print(f"❌ Error during inference: {e}")

# ==== STEP 4: Save Results ====
OUTPUT_FILE = "output_llama2.json"
try:
    with open(OUTPUT_FILE, "w") as f:
        json.dump(responses, f, indent=4)
    print(f"📂 Results saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"❌ Error saving results: {e}")

print("🎉 === Inference Completed ===")



