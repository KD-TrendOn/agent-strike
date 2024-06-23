from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()


OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')


better_llm = ChatOpenAI(
        temperature=0,
        model="anthropic/claude-3.5-sonnet",#openai/gpt-4o
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )
