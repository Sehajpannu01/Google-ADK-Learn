# agents/web_research_agent.py
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import litellm
from google.adk.tools import google_search

import os
load_dotenv()

litellm.use_litellm_proxy = True
 
lite_llm_model = LiteLlm(
    model="gemini-2.5-flash",
    api_base=os.getenv("LITELLM_PROXY_API_BASE"),
    api_key=os.getenv("LITELLM_PROXY_GEMINI_API_KEY")    
)

def fetch_and_summarize(topic: str) -> dict:
    results = google_search(topic)

    if not results:
        return {"status": "error", "summary": ""}

    text = ""
    if isinstance(results, list):
        text = results[0].get("snippet", "")[:500]

    return {
        "status": "success",
        "summary": text
    }

web_research_agent = Agent(
    name="web_research_agent",
    model=lite_llm_model,
    description="Fetches data from the web and summarizes it",
    instruction="Fetch information from the web and produce a concise factual summary.",
    tools=[fetch_and_summarize],
)
