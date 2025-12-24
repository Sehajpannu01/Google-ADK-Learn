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

def get_current_time():
    import datetime
    return datetime.datetime.now().isoformat()


root_agent = Agent(
    name="tool_agent",
    model=lite_llm_model,
    description="Tool Agent",
    instruction= '''You are helpful assistant that can use following tools:
    -google_search 
    ''',
    tools=[google_search],
    # tools=[get_current_time],
)


def run_agent_prompt(prompt: str):
    """
    Standard wrapper for test harness:
    Accepts prompt string, runs the agent, returns the agent response object
    (or text) — whatever your agent returns normally.
    """
    import asyncio
    import uuid
    from google.genai import types as genai_types
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.memory import InMemoryMemoryService
    from google.adk.utils.context_utils import Aclosing
    
    async def _run():
        # Setup services
        session_service = InMemorySessionService()
        artifact_service = InMemoryArtifactService()
        memory_service = InMemoryMemoryService()
        
        # Create session
        app_name = "test_app"
        user_id = "test_user"
        session_id = str(uuid.uuid4())
        
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            state={},
            session_id=session_id
        )
        
        # Create Runner
        runner = Runner(
            app_name=app_name,
            agent=root_agent,
            artifact_service=artifact_service,
            session_service=session_service,
            memory_service=memory_service
        )
        
        # Create user content
        user_content = genai_types.Content(
            parts=[genai_types.Part(text=prompt)],
            role='user'
        )
        
        # Run the agent
        final_response = None
        async with Aclosing(runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content
        )) as agen:
            async for event in agen:
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content
        
        return final_response
    
    return asyncio.run(_run())