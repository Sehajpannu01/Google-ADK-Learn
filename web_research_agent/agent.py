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

def fetch_and_summarize(query: str) -> dict:
    """
    Non-conversational agent.
    Trigger → fetch → summarize → return data
    """
    try:
        results = google_search(query)
    except Exception as e:
        return {"status": "error", "summary": ""}

    summary = ""
    if isinstance(results, list) and results:
        summary = results[0].get("snippet", "")[:400]

    return {
        "status": "success",
        "summary": summary
    }

root_agent = Agent(
    name="web_research_agent",
    model=lite_llm_model,
    description="Fetches data from web and summarizes",
    instruction="Fetch information from web and return factual summary only.",
    tools=[fetch_and_summarize],
)
