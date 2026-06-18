from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import os


class source(BaseModel):
    """schema for a source used by agent."""
    
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """schema for agent response."""
    
    answer: str = Field(description="Content of the agent's response")
    sources: List[source] = Field(default_factory=list, description="List of sources used by the agent")

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages": HumanMessage(content="search of three job posting for an AI Engineer using Lang chain in the Hyderabad area on LinkedIn and list their details")})
    print(result)

if __name__ == "__main__":
    main()