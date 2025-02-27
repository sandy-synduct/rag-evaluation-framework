import json
import os
import requests

# API endpoint
API_URL = "http://127.0.0.1:8000/chat"

# Load queries from the JSON file
with open("eval_questions.json", "r", encoding="utf-8") as f:
    questions_data = json.load(f)

# Iterate over each query and send it sequentially
for idx, item in enumerate(questions_data["evaluation_questions"]):
    query_text = item["question"]
    
    print(f"Sending Query {idx + 1}/{len(questions_data['evaluation_questions'])}: {query_text}")

    # Prepare payload
    payload = {
        "query": query_text,
        "max_results": 10  # Adjust based on your API requirements
    }
    try:
        # Send request
        response = requests.post(API_URL, json=payload)
        # print(response.json())
        response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)

    except requests.exceptions.RequestException as e:
        print(f"Error sending query {idx + 1}: {e}")

    break





