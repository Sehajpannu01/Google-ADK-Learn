from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import litellm
 
import os
load_dotenv()
 
litellm.use_litellm_proxy = True
 
lite_llm_model = LiteLlm(
    model="gemini-2.0-flash",
    api_base=os.getenv("LITELLM_PROXY_API_BASE"),
    api_key=os.getenv("LITELLM_PROXY_GEMINI_API_KEY")    
)
root_agent = Agent(
    name="greeting_agent",
    model=lite_llm_model,
    description="Greeting Agent",
    instruction= '''You are a friendly assistant that greets people. Ask for their name and greet them personally.''',
)