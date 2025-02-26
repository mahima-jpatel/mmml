import json
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from datasets import load_dataset
from tqdm import tqdm

# Model and dataset details
MODEL_ID = "Salesforce/instructblip-vicuna-7b"
DATASET_NAME = "AI4Math/MathVista"
OUTPUT_FILE = "testinstructblip_results.json"

# Initialize model and processor
#model = InstructBlipForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map={"": "cuda"}  # Explicit device mapping
)
print("model assigned")
#processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
processor = InstructBlipProcessor.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset(DATASET_NAME, split="testmini")
print("datset loaded")
# Store results in a dictionary
results = {}

for i, sample in tqdm(enumerate(dataset), total=len(dataset)):

    # Load image
    # image_url = sample.get("image", "")
    # if image_url:
    #     image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    # else:
    #     image = None
    print(i)
    image = sample["decoded_image"].convert("RGB")
    print(image)
    # Define prompt
    prompt = sample.get("query", sample["query"])
    #formatted_prompt = f"Provide a detailed and accurate answer:\n\nQuestion: {sample['question']}\n\nAnswer:"
    #print(formatted_prompt)
    # Process input
    # if image:
    #     inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    # else:
    #     inputs = processor(text=prompt, return_tensors="pt").to(device)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        do_sample=True,
        num_beams=5,
        max_new_tokens=1024,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    # Store result
    results[str(sample["pid"])] = {
        "question": sample["question"],
        "image": sample.get("image", "none"),
        "choices": sample.get("choices", "none"),
        "unit": sample.get("unit", "none"),
        "precision": sample.get("precision", 1),
        "answer": sample.get("answer", "none"),
        "question_type": sample.get("question_type", "free_form"),
        "answer_type": sample.get("answer_type", "text"),
        "pid": sample.get("pid", str(i + 1)),
        "metadata": sample.get("metadata", {}),
        "query": prompt,
        "response": response,
    }
    
    # Save after each iteration
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if i >= 9: 
        break

