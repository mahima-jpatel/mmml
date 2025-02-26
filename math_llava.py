from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from datasets import load_dataset
import torch
import json

# Updated Math-LLaVA Model ID
MODEL_ID = "Zhiqiang007/Math-LLaVA"
DATASET_NAME = "AI4Math/MathVista"
OUTPUT_FILE = "mathvista_results_mathllava.json"

# Initialize model with optimized precision
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

# Ensure correct tokenization and padding
processor.tokenizer.padding_side = "left"

# Load test subset from MathVista dataset
dataset = load_dataset(DATASET_NAME, split="testmini")

from tqdm import tqdm
results = {}

for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
    try:
        # Decode image correctly
        image = sample["decoded_image"].convert("RGB")

        # Format conversation with Math-LLaVA structure
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample["query"]}
            ]
        }]
        
        # Apply correct chat template for Math-LLaVA
        prompt = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )

        # Convert input into model-friendly format
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(model.device, torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        # Generate output with Math-LLaVA
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Adjusted for mathematical answers
            temperature=0.7,  # Balanced creativity and correctness
            do_sample=True  # Enables non-beam search sampling
        )
        
        # Decode and clean the response
        response = processor.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()

        # Store results
        results[str(sample["pid"])] = {
            "pid": sample.get("pid", str(i + 1)),
            "question": sample["question"],
            "image": sample.get("image", "none"),
            "choices": sample.get("choices", "none"),
            "unit": sample.get("unit", "none"),
            "precision": sample.get("precision", 1),
            "answer": sample.get("answer", "none"),
            "question_type": sample.get("question_type", "free_form"),
            "answer_type": sample.get("answer_type", "text"),
            "metadata": sample.get("metadata", {}),
            "query": sample.get("query", prompt),
            "response": response
        }
        
        # Save results after each sample (for safety)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"Error processing {sample.get('pid')}: {str(e)}")

    # Optional: Limit dataset for testing
    if i >= 9:
        break
