import google.generativeai as genai
from openai import OpenAI
import streamlit as st
import os

class LLMService:
    def __init__(self, gemini_api_key: str = None, openai_api_key: str = None, model_name: str = "gemini-2.0-flash-exp"):
        self.provider = model_name
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key

        if self.provider == "gemini-2.0-flash-exp":
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        elif self.provider == "gpt-4o-mini":
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.client = OpenAI()
        else:
            raise ValueError("Unsupported provider. Choose 'Gemini' or 'OpenAI'.")

    def create_prompt(self, abstract: str) -> str:
        """Generate a structured prompt for summarizing a single abstract."""
        return f"""
                Task: Summarize the following clinical abstract accurately within 300 words while preserving all key information. Do not introduce new information, make assumptions, or hallucinate facts.
            Always include clinical medical terms or keywords that could be potential MeSH Terms
            Instructions:
                •	Extract all relevant details including background, objective, methods, results, and conclusions.
            •	Maintain the same meaning and intent as the original abstract.
            •	Avoid unnecessary repetition or filler words.
            •	Ensure that any numerical data, medical terms, and key findings are retained and accurately conveyed.
            •	Maintain clinical precision and do not infer beyond what is stated.
            •	If the abstract contains unclear statements, phrase them as presented without guessing or modifying the meaning.
            •	Strictly adhere to the 300-word limit.

            Abstract:
        {abstract}

            Summary:
                
        """

    def summarize_abstract(self, abstract: str) -> str:
        """Summarize a single abstract using the selected LLM provider."""
        try:
            prompt = self.create_prompt(abstract)
            
            if self.provider == "gemini-2.0-flash-exp":
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.4,
                        'max_output_tokens': 500,
                        'top_k': 50,
                        'top_p': 0.8,
                    }
                )
                return response.text.strip() if response.text else "No summary available."

            elif self.provider == "gpt-4o-mini":
                response = self.client.chat.completions.create(
                    model=self.provider,
                    messages=[{"role": "system", "content": "You are a medical expert."},
                              {"role": "user", "content": prompt}],
                    temperature=0.4
                ) 
                # print(response)
                return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {e}"

    def summarize_abstracts(self, abstracts: list) -> dict:
        """Summarize each abstract using the chosen provider."""
        summaries = []
        for i, abstract in enumerate(abstracts):
            summary = self.summarize_abstract(abstract)
            summaries.append(summary)
        return summaries