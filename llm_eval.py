import google.generativeai as genai
import os, json,re
import subprocess
from generate_markdown import json_to_markdown_evaluation
import re
import json
import markdown
import pdfkit
import pypandoc

genai.configure(api_key="AIzaSyBMbbqL-DgamRfyEGU5GB02ciGnlQh_zGY")
model = genai.GenerativeModel('gemini-2.0-flash-exp')
with open('eval_prompt.txt','r') as f:
    prompt = f.read()

eval_folder = '/Users/sandhanakrishnan/merged_drinfo/services/ai_summary/eval_folder'
json_files = [os.path.join(eval_folder,file) for file in os.listdir(eval_folder)]
for json_file in json_files:
    with open(json_file,'r') as f:
        data = json.load(f)
        search_query= data['search_query']
        json_string= json.dumps(data, indent=4)
        prompt_with_query = f"{prompt.format(user_query_placeholder=search_query,json_data_placeholder=json_string)}"
        # print(prompt_with_query)
        response = model.generate_content(
                        prompt_with_query,
                        generation_config={
                            'temperature': 0.4,  # Lower temperature for more focused medical responses
                            'max_output_tokens': 8000,  # Increased for comprehensive medical answers
                            'top_k': 70,
                            'top_p': 0.8,

                        }
                    )
    # print(response.text)
    safe_query = re.sub(r'[^\w\s-]', '', search_query).strip().replace(" ", "_")
    json_match = re.search(r'`json\s*([\s\S]*?)\s*`', response.text)
    if json_match:
                json_str_extracted = json_match.group(1)
                evaluation_json = json.loads(json_str_extracted)
                evaluation_json['search_results'] = data['search_results']
                evaluation_json['citations'] = data['citations']
                evaluation_json['summary'] = data['summary']
                evaluation_json['search_query'] = data['search_query']
                markdown_txt =  json_to_markdown_evaluation(evaluation_json)
                markdown_path = f'eval_report/{safe_query}.md'
                with open(markdown_path, "w", encoding="utf-8") as md_file:
                        md_file.write(markdown_txt)

    # Convert Markdown to PDF
                # pdf_path = f'eval_report/{safe_query}.pdf'
                # pypandoc.convert_file(markdown_path, 'pdf', outputfile=pdf_path)

    break


# docker run --rm \
#     -v $PWD:/app \
#     -u "$(id -u):$(id -g)" \
#     jmaupetit/md2pdf --css styles.css What_are_the_diagnostic_criteria_for_systemic_lupus_erythematosus_SLE_according_to_the_2019_EULARACR_classification.md test.pdf