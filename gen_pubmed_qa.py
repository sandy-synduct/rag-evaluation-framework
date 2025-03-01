import json
import os
import requests
from datasets import load_dataset
import google.generativeai as genai
import os, json,re
import re
import json



ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
API_URL = "http://127.0.0.1:8000/chat"


start_idx = 11  
end_idx = 100    

for idx in range(start_idx, min(end_idx, len(ds['train']))):
    eval = ds['train'][idx]
    query_text = eval['question']

    print(f"Sending Query {idx + 1}/{len(ds['train'])}: {query_text}")

    payload = {
        "query": query_text,
        "max_results": 10  # Adjust based on your API requirements
    }
    try:
        # Send request
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  
        print(f"Response for Query {idx + 1}: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending query {idx + 1}: {e}")





