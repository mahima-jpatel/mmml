import json
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Model and dataset details
MODEL_ID = "Salesforce/instructblip-vicuna-7b"
DATASET_NAME = "AI4Math/MathVista"
OUTPUT_FILE = "testinstructblip_results.json"

# Initialize model and processor
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map={"": "cuda"}
)
processor = InstructBlipProcessor.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset(DATASET_NAME, split="testmini")
print("Dataset loaded")

# Store results in a dictionary
results = {}

for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
    #print(f"Processing sample {i}")
    image = sample["decoded_image"].convert("RGB")
    print(sample["decoded_image"])
    print(image)
    prompt = f"Please look at the question: {sample['query']}. Look at the image and reason the solution and return final answer. Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16).to(device)

# autoregressively generate an answer
    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    outputs[outputs == 0] = 2 # this line can be removed once https://github.com/huggingface/transformers/pull/24492 is  fixed
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    print(processor.batch_decode(outputs, skip_special_tokens=True))
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
