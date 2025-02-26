import json

# Load JSON file
with open("output_llama2.json", "r") as f:
    data = json.load(f)  # Load JSON as a Python list

# Print first dictionary in JSON format
if isinstance(data, list) and len(data) > 0:
    print(json.dumps(data[0], indent=4))  # Pretty-print first dict
else:
    print("❌ JSON file is empty or not a list.")
