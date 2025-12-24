# agents/data_generation_agent.py
from google.adk.agents import Agent

def generate_report(topic: str) -> dict:
    return {
        "status": "success",
        "data": {
            "topic": topic,
            "metrics": {
                "confidence": 0.92,
                "coverage": "high",
                "risk_level": "low"
            }
        }
    }

data_generation_agent = Agent(
    name="data_generation_agent",
    model="gemini-2.0-flash",
    description="Generates structured data on trigger",
    instruction="Generate structured JSON data when triggered.",
    tools=[generate_report],
)
