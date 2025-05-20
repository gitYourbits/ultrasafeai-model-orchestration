import os
from openai import OpenAI
from typing import Dict
import logging

class ReportGeneratorAgent:
    """
    Agent to generate a human-readable summary from key financial metrics and trends.
    """
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_report(self, metrics: Dict) -> str:
        self.logger.info("Starting report generation.")
        prompt = (
            """
            You are a financial report writer. Given the following extracted financial metrics and trends (in JSON), write a concise, clear, and professional summary suitable for a business executive. Highlight the most important findings and trends.
            
            Financial Metrics and Trends:
            """
            + str(metrics)
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            self.logger.info("Report generated successfully.")
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise

if __name__ == "__main__":
    import sys
    import json
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <metrics_json_file>")
        exit(1)
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        print("Please set OPENAI_API_KEY and OPENAI_BASE_URL environment variables.")
        exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        metrics = json.load(f)
    agent = ReportGeneratorAgent(api_key, base_url)
    report = agent.generate_report(metrics)
    print(report) 