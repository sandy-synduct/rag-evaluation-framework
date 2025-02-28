import re 


def json_to_markdown_evaluation(json_data):
    """
    Converts the JSON evaluation output to a Markdown document with a highlighted score table,
    summary display with spaced and linked citations (including lists like [1, 2]),
    evaluation sections with spaced and linked citations, and linked citations section.

    Args:
        json_data (dict): The JSON object containing the evaluation results, including 'summary' and 'search_query'.

    Returns:
        str: A Markdown formatted string representing the evaluation.
    """

    markdown_output = f"# Evaluation for Query: {json_data['search_query']}\n\n" # Added search query as top heading

    # 1. Score Table - Highlighted with Colors and Borders
    markdown_output += "## Evaluation Scores\n\n"

    # Table with colored headers
    markdown_output += """<table>
    <tr>
        <th style="background-color:#ff4d4d; color:black; padding:6px; border: 1px solid black;">Metric</th>
        <th style="background-color:#4CAF50; color:white; padding:6px; border: 1px solid black;">Score</th>
    </tr>
    """

    # Hallucination Score
    hallucination_eval = json_data["hallucination_evaluation"]
    score_description_hallucination = ["None", "Minor", "Moderate", "High"][int(hallucination_eval['hallucination_score'])]
    markdown_output += f"""<tr>
        <td style="border: 1px solid black; padding:6px;"><b>Hallucination Score</b></td>
        <td style="border: 1px solid black; padding:6px;"><b>{hallucination_eval['hallucination_score']} - {score_description_hallucination}</b></td>
    </tr>
    """

    # Summary Quality Score
    summary_quality_eval = json_data["summary_quality_evaluation"]
    markdown_output += f"""<tr>
        <td style="border: 1px solid black; padding:6px;"><b>Summary Quality Score</b></td>
        <td style="border: 1px solid black; padding:6px;"><b>{summary_quality_eval['summary_quality_score']}%</b></td>
    </tr>
    """

    # Answer Relevancy Score
    relevancy_eval = json_data["answer_relevancy_evaluation"]
    score_description_relevancy = ["Not Relevant", "Partially Relevant", "Relevant", "Highly Relevant"][int(relevancy_eval['relevancy_score'])]
    markdown_output += f"""<tr>
        <td style="border: 1px solid black; padding:6px;"><b>Answer Relevancy Score</b></td>
        <td style="border: 1px solid black; padding:6px;"><b>{relevancy_eval['relevancy_score']} - {score_description_relevancy}</b></td>
    </tr>
    </table>
    """

    markdown_output += "\n---\n\n"  # Separator line


    # 2. Display Summary from Search Results with Spaced and Linked Citations (Handling lists like [1, 2])
    markdown_output += "## Generated Summary\n\n"
    summary_text = json_data['summary']

    def replace_citation_list(match, citations_data): # Pass citations_data to the function
        citation_list_str = match.group(0) # e.g., "[1, 2]"
        citation_numbers = re.findall(r'\d+', citation_list_str) # Extract numbers: ['1', '2']
        markdown_links = []
        for citation_number in citation_numbers:
            if citation_number in citations_data: # Check if citation exists
                markdown_links.append(f"[<sup>{citation_number}</sup>](#citation-{citation_number})") # Create link for each
            else:
                markdown_links.append(f"<sup>{citation_number}</sup>") # If citation doesn't exist, just use superscript
        return " ".join(markdown_links) # Join links with spaces


    # Use regex to find citation lists like [1], [1, 2], [2,3,8] and replace them in summary
    summary_text = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', lambda match: replace_citation_list(match, json_data["citations"]), summary_text)
    markdown_output += f"{summary_text}\n\n"
    markdown_output += "---\n\n" # Separator line


    markdown_output += "## Citations\n\n"
    citations = json_data["citations"]
    for index, citation_item in enumerate(citations.items()):
        citation_num_key, citation_data = citation_item
        citation_number = str(citation_num_key) # Use citation_num_key directly as citation number for display order
        markdown_output += f"<a name=\"citation-{citation_number}\"></a>**<sup>[{citation_number}]</sup>** " # Anchor and superscript
        markdown_output += f"[{citation_data['text']}]({citation_data['url']})\n\n" # Using text from citation data


    
    # 3. Evaluation Report

    # 3.1. Hallucination Evaluation Report
    markdown_output += "## Hallucination Evaluation Report\n\n"
    markdown_output += f"**Hallucination Score:** {hallucination_eval['hallucination_score']} - {score_description_hallucination}\n\n"
    markdown_output += f"**Justification:** "
    justification_text_hallucination = hallucination_eval['hallucination_justification']
    justification_text_hallucination = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', lambda match: replace_citation_list(match, json_data["citations"]), justification_text_hallucination) # Apply citation linking to justification
    markdown_output += f"{justification_text_hallucination}\n\n"
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
                justification_text = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', lambda match: replace_citation_list(match, json_data["citations"]), justification_text) # Apply citation linking to justification
                markdown_output += f"    * **Justification:** {justification_text}\n\n"
            else:
                markdown_output += "\n" # Add newline if no justification
        markdown_output += "\n" # Add extra newline between search result sections
    markdown_output += "---\n\n" # Separator line


    # 3.3. Answer Relevancy Evaluation Report
    markdown_output += "## Answer Relevancy Evaluation Report\n\n"
    markdown_output += f"**Relevancy Score:** {relevancy_eval['relevancy_score']} - {score_description_relevancy}\n\n"
    markdown_output += f"**Justification:** "
    justification_text_relevancy = relevancy_eval['relevancy_justification']
    justification_text_relevancy = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', lambda match: replace_citation_list(match, json_data["citations"]), justification_text_relevancy) # Apply citation linking to justification
    markdown_output += f"{justification_text_relevancy}\n\n"
    markdown_output += "**Relevant Search Result URLs:**\n"
    for url in relevancy_eval["relevant_search_result_urls"]:
        markdown_output += f"* [{url}]({url})\n"
    markdown_output += "\n---\n\n" # Separator line


    return markdown_output


    # 4. Citations Section
    


