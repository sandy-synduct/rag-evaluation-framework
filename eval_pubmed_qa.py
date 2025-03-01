import json
import os
import csv
import re
import requests
from datasets import load_dataset
import google.generativeai as genai
from datasets import Dataset
from ragas.metrics import answer_similarity, answer_correctness
from ragas import evaluate

from tqdm import tqdm

with open('eval_prompt_pubmed.txt', 'r') as f:
    prompt = f.read()

# Path to evaluation folder
eval_folder = '/Users/sandhanakrishnan/merged_drinfo/services/ai_summary/eval_folder'

# Configure GenAI Model
genai.configure(api_key="AIzaSyBb3G1jsuU9swQMumnZv7EkU2zOcJkwha0")
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Load dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

# Define file paths
json_output_file = "evaluation_results.json"
csv_output_file = "evaluation_results.csv"  

# Ensure the eval folder exists
os.makedirs(eval_folder, exist_ok=True)

# Initialize storage containers
all_evaluations = []  # List for JSON storage
processed_files = set()  # Track processed files
csv_headers = set()  # Track all possible keys dynamically

# Load existing evaluations from JSON to resume
if os.path.exists(json_output_file):
    with open(json_output_file, "r") as f:
        try:
            all_evaluations = json.load(f)
            processed_files = {entry["search_query"] for entry in all_evaluations}  # Track completed queries
        except json.JSONDecodeError:
            all_evaluations = []
            processed_files = set()

# Load existing CSV to update headers
if os.path.exists(csv_output_file):
    with open(csv_output_file, "r") as f:
        reader = csv.DictReader(f)
        csv_headers.update(reader.fieldnames)  # Load headers if file exists

# Get all JSON files sorted
json_files = sorted([os.path.join(eval_folder, file) for file in os.listdir(eval_folder) if file.endswith(".json")])

# Initialize progress bar
with tqdm(total=len(json_files), desc="Processing Files") as pbar:
    for idx, json_file in enumerate(json_files):
        # Load JSON file
        with open(json_file, "r") as f:
            data = json.load(f)

        search_query = data["search_query"]

        # Skip if already processed
        if search_query in processed_files:
            pbar.update(1)
            continue

        print(f"Processing: {json_file}")

        # Get ground truth answer
        eval = ds["train"][idx]

        json_string = json.dumps(data, indent=4)
        prompt_with_query = f"{prompt.format(user_query_placeholder=search_query, json_data_placeholder=json_string)}"

        # Generate response
        response = model.generate_content(
            prompt_with_query,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 8000,
                "top_k": 70,
                "top_p": 0.8,
            }
        )

        # Extract JSON response
        json_match = re.search(r"`json\s*([\s\S]*?)\s*`", response.text)

        if json_match:
            json_str_extracted = json_match.group(1)
            evaluation_json = json.loads(json_str_extracted)

            # Add additional fields
            evaluation_json["summary"] = data["summary"]
            evaluation_json["search_query"] = search_query
            evaluation_json["ground_truth_answer"] = eval["long_answer"]
            evaluation_json["ground_truth_final_decision"] = eval["final_decision"]

            # Compute RAGAS metrics
            data_samples = {
                "question": [search_query],
                "answer": [data["summary"]],
                "ground_truth": [eval["long_answer"]],
            }
            dataset = Dataset.from_dict(data_samples)
            score = evaluate(dataset, metrics=[answer_similarity])

            evaluation_json["ragas_semantic_similarity"] = score["semantic_similarity"][0]

            # Append to JSON list
            all_evaluations.append(evaluation_json)
            processed_files.add(search_query)  # Mark as processed

            # Save to JSON after each iteration to prevent data loss
            with open(json_output_file, "w") as file:
                json.dump(all_evaluations, file, indent=4)

            # Update CSV headers dynamically
            csv_headers.update(evaluation_json.keys())

            # Append to CSV
            file_exists = os.path.exists(csv_output_file)
            with open(csv_output_file, "a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(csv_headers))

                # Write header if file does not exist
                if not file_exists:
                    writer.writeheader()

                # Write row
                writer.writerow(evaluation_json)

            print(f"Processed and saved evaluation for {json_file}")

        # Update progress bar
        pbar.update(1)

print(f"Saved all evaluations to {json_output_file}")
print(f"Saved all evaluations to {csv_output_file}")