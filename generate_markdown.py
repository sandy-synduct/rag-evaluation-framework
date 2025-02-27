import re # Import the regular expression library


def json_to_markdown_evaluation(json_data):
    """
    Converts the JSON evaluation output to a Markdown document with a score table,
    summary display with linked citations (including lists like [1, 2]),
    evaluation sections, and linked citations.

    Args:
        json_data (dict): The JSON object containing the evaluation results, including 'summary' and 'search_query'.

    Returns:
        str: A Markdown formatted string representing the evaluation.
    """

    markdown_output = f"# Evaluation for Query: {json_data['search_query']}\n\n" # Added search query as top heading

    # 1. Score Table
    markdown_output += "## Evaluation Scores\n\n"
    markdown_output += "| Metric                  | Score         |\n"
    markdown_output += "| ----------------------- | ------------- |\n"

    hallucination_eval = json_data["hallucination_evaluation"]
    score_description_hallucination = ""
    score_h = int(hallucination_eval['hallucination_score'])
    if score_h == 0:
        score_description_hallucination = "None"
    elif score_h == 1:
        score_description_hallucination = "Minor"
    elif score_h == 2:
        score_description_hallucination = "Moderate"
    elif score_h == 3:
        score_description_hallucination = "High"
    markdown_output += f"| **Hallucination Score** | **{hallucination_eval['hallucination_score']} - {score_description_hallucination}** |\n"

    summary_quality_eval = json_data["summary_quality_evaluation"]
    markdown_output += f"| **Summary Quality Score** | **{summary_quality_eval['summary_quality_score']}%**         |\n"

    relevancy_eval = json_data["answer_relevancy_evaluation"]
    score_description_relevancy = ""
    rel_score = int(relevancy_eval['relevancy_score'])
    if rel_score == 3:
        score_description_relevancy = "Highly Relevant"
    elif rel_score == 2:
        score_description_relevancy = "Relevant"
    elif rel_score == 1:
        score_description_relevancy = "Partially Relevant"
    elif rel_score == 0:
        score_description_relevancy = "Not Relevant"
    markdown_output += f"| **Answer Relevancy Score**| **{relevancy_eval['relevancy_score']} - {score_description_relevancy}** |\n"
    markdown_output += "\n---\n\n" # Separator line


    # 2. Display Summary from Search Results with Linked Citations (Handling lists like [1, 2])
    markdown_output += "## Generated Summary\n\n"
    summary_text = json_data['summary']

    def replace_citation_list(match):
        citation_list_str = match.group(0) # e.g., "[1, 2]"
        citation_numbers = re.findall(r'\d+', citation_list_str) # Extract numbers: ['1', '2']
        markdown_links = []
        for citation_number in citation_numbers:
            markdown_links.append(f"[<sup>{citation_number}</sup>](#citation-{citation_number})") # Create link for each
        return "".join(markdown_links) # Join links together

    # Use regex to find citation lists like [1], [1, 2], [2,3,8] and replace them
    summary_text = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', replace_citation_list, summary_text)
    markdown_output += f"{summary_text}\n\n"
    markdown_output += "---\n\n" # Separator line


    # 3. Evaluation Report

    # 3.1. Hallucination Evaluation Report
    markdown_output += "## Hallucination Evaluation Report\n\n"
    markdown_output += f"**Hallucination Score:** {hallucination_eval['hallucination_score']} - {score_description_hallucination}\n\n"
    markdown_output += f"**Justification:** {hallucination_eval['hallucination_justification']}\n\n"
    markdown_output += "---\n\n" # Separator line


    # 3.2. Summary Quality Evaluation Report
    markdown_output += "## Summary Quality Evaluation Report\n\n"
    markdown_output += f"**Summary Quality Score:** {summary_quality_eval['summary_quality_score']}%\n\n"
    markdown_output += "### Questions and Answers Details:\n\n"

    for qa_section in summary_quality_eval["questions_and_answers"]:
        text_url = qa_section["text_url"]
        title = ""
        for index, result in enumerate(json_data["search_results"]): # Enumerate for citation index
            if result["url"] == text_url:
                citation_number = str(index + 1) # Citation number is index + 1
                title = result["metadata"].get("title", f"Search Result {citation_number}")
                break
        else: # If no title found from search results, use URL last part as title
            title_from_url = text_url.split('/')[-1] if text_url.split('/')[-1] else "No Title Found"
            title = title_from_url if title_from_url != "No Title Found" else f"Search Result URL {citation_number if 'citation_number' in locals() else 'Unknown'}"


        markdown_output += f"#### Search Result: [{title}]({text_url}) [[{citation_number}]](#citation-{citation_number})\n" # Add citation link
        markdown_output += f"**URL:** [{text_url}]({text_url})\n\n"

        for q_and_a in qa_section["questions"]:
            answered_status = "Yes" if q_and_a["is_answered"] else "No"
            markdown_output += f"* **Question:** {q_and_a['question']}\n"
            markdown_output += f"    * **Answered:** {answered_status}\n"
            if q_and_a.get("answer_justification"): # Justification is optional
                justification_text = q_and_a['answer_justification']
                # Replace citation numbers in justification with Markdown links
                for citation_num_key, citation_data in json_data["citations"].items():
                     citation_placeholder = f"[{citation_num_key}]"
                     markdown_citation_link = f"[<sup>{citation_num_key}</sup>](#citation-{citation_num_key})"
                     justification_text = justification_text.replace(citation_placeholder, markdown_citation_link)

                markdown_output += f"    * **Justification:** {justification_text}\n\n"
            else:
                markdown_output += "\n" # Add newline if no justification
        markdown_output += "\n" # Add extra newline between search result sections
    markdown_output += "---\n\n" # Separator line


    # 3.3. Answer Relevancy Evaluation Report
    markdown_output += "## Answer Relevancy Evaluation Report\n\n"
    markdown_output += f"**Relevancy Score:** {relevancy_eval['relevancy_score']} - {score_description_relevancy}\n\n"
    markdown_output += f"**Justification:** {relevancy_eval['relevancy_justification']}\n\n"
    markdown_output += "**Relevant Search Result URLs:**\n"
    for url in relevancy_eval["relevant_search_result_urls"]:
        markdown_output += f"* [{url}]({url})\n"
    markdown_output += "\n---\n\n" # Separator line


    # 4. Citations Section
    markdown_output += "## Citations\n\n"
    citations = json_data["citations"]
    for index, citation_item in enumerate(citations.items()):
        citation_num_key, citation_data = citation_item
        citation_number = str(index + 1) # Citation number is index + 1 for display order
        markdown_output += f"<a name=\"citation-{citation_number}\"></a>**<sup>[{citation_number}]</sup>** " # Anchor and superscript
        markdown_output += f"[{citation_data['text']}]({citation_data['url']})\n\n"


    return markdown_output


