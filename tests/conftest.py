import os
import litellm
from deepeval.models import LiteLLMModel
from dotenv import load_dotenv

load_dotenv()

litellm.use_litellm_proxy = True

EVAL_MODEL = LiteLLMModel(
    model="ai-studio-gemini-2.0-flash",   # works already
)
