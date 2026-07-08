import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
load_dotenv()
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool function that searches for the weather in Tokyo and returns a string with the result.
    Args:
        query (str): The search query.
    Returns:
        str: The search result.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

tools = [TavilySearch(), search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from agent!")
    result = agent.invoke({"messages": [HumanMessage(content="Poszukaj na linkedin  5 ogłoszeń o pracę jako langchail developer lub podobne")]})
    print(result)

if __name__ == "__main__":
    main()
