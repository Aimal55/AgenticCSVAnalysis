import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import HuggingFaceHub
import pandas as pd

load_dotenv()

def create_agent_from_csv(csv_path):
    huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 500},
        huggingfacehub_api_token=huggingface_token,
    )

    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
    agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=False,
    agent_type="openai-tools",  # optional, keeps it updated with LangChain’s internal agent type
    allow_dangerous_code=True   # ✅ this enables Python code execution
    )
    return agent