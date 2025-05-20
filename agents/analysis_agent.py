import os
from openai import OpenAI
from typing import Dict, Any
import logging

class AnalysisAgent:
    """
    Agent to identify key financial metrics and trends from extracted text.
    """
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self, document_text: str) -> Dict[str, Any]:
        self.logger.info("Starting analysis of document text.")
        prompt = (
            """
            You are a financial analyst. Extract the following key metrics from the provided financial report text:
            - Revenue
            - Net Income
            - Operating Expenses
            - Gross Profit
            - Year-over-Year Growth
            - Any notable financial trends
            Return the results as a JSON object with clear keys.
            
            Financial Report Text:
            """
            + document_text
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            import json
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            self.logger.info("Analysis completed successfully.")
            return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {"error": f"Failed to parse response: {str(e)}", "raw_response": response.choices[0].message.content}

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python analysis_agent.py <text_file>")
        exit(1)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        print("Please set OPENAI_API_KEY and OPENAI_BASE_URL environment variables.")
        exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text = f.read()
    agent = AnalysisAgent(api_key, base_url)
    result = agent.analyze(text)
    print(result) 