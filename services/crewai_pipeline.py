import json
from datetime import datetime

import markdown2
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import tool
from duckduckgo_search import DDGS
import yfinance as yf
from curl_cffi import requests

from flask import current_app

# Simple tools implemented via @tool for CrewAI agents
@tool("DuckDuckGo Search")
def search_tool(search_query: str) -> str:
    """Search the web for information on a given topic."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))
            if results:
                return "\n\n".join([
                    f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results
                ])
            return "No results found"
    except Exception as e:
        return f"Search failed: {e}"

@tool("get current stock price")
def get_curr_stock_price(symbol: str) -> str:
    """Fetch the current stock price for the given symbol."""
    try:
        session = requests.Session(impersonate="chrome")
        stock = yf.Ticker(symbol, session=session)
        info = stock.info or {}
        current_price = info.get("regularMarketPrice", info.get("currentPrice"))
        if current_price is None:
            return f"could not fetch current price for {symbol}"
        return f"{float(current_price):.2f}"
    except Exception as e:
        return f"error fetching information for {symbol}: {e}"

@tool("get company info")
def get_company_info(symbol: str) -> str:
    """Retrieve detailed company profile information for the given stock symbol."""
    try:
        session = requests.Session(impersonate="chrome")
        info = yf.Ticker(symbol, session=session).info
        if not info:
            return f"Could not fetch company info for {symbol}"
        cleaned = {
            "Name": info.get("shortName"),
            "Symbol": info.get("symbol"),
            "Current Stock Price": f"{info.get('regularMarketPrice', info.get('currentPrice'))} {info.get('currency')}",
            "Market Cap": f"{info.get('marketCap', info.get('enterpriseValue'))} {info.get('currency')}",
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Country": info.get("country"),
            "P/E Ratio": info.get("trailingPE"),
            "52 Week Low": info.get("fiftyTwoWeekLow"),
            "52 Week High": info.get("fiftyTwoWeekHigh"),
        }
        return json.dumps(cleaned, indent=2)
    except Exception as e:
        return f"could not fetch company profile information on {symbol}: {e}"

@tool("get income statements")
def get_income_statements(symbol: str) -> str:
    """Fetch the income statement data for the specified stock symbol."""
    try:
        session = requests.Session(impersonate="chrome")
        stock = yf.Ticker(symbol, session=session)
        financials = stock.financials
        if financials is None or getattr(financials, "empty", True):
            return f"No income statement data available for {symbol}"
        return financials.head(10).to_json(orient="index")
    except Exception as e:
        return f"error fetching income statements for {symbol}: {e}"

def get_llm():
    """Create an LLM instance configured for OpenRouter."""
    api_key = current_app.config.get("OPENROUTER_API_KEY")
    base_url = current_app.config.get("OPENROUTER_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Put it in your .env as OPENROUTER_API_KEY or openrouter_api_key.")
    return LLM(
        model="openrouter/deepseek/deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
    )

def run_investment_crew(stock_symbol: str):
    """Runs the CrewAI analysis for a given stock symbol and returns (analysis_html, recommendation_html)."""
    Today = datetime.now().strftime("%d-%b-%Y")
    llm = get_llm()

    news_info_explorer = Agent(
        role="News Researcher",
        goal=f"Find latest news about {stock_symbol}",
        llm=llm,
        verbose=True,
        backstory=f"You gather company news. Today is {Today}",
        tools=[search_tool],
        max_iter=2,
        allow_delegation=False,
    )

    data_explorer = Agent(
        role="Financial Data Researcher",
        goal=f"Get financial data for {stock_symbol}",
        llm=llm,
        verbose=True,
        backstory=f"You gather financial data. Today is {Today}",
        tools=[get_company_info, get_income_statements],
        max_iter=2,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Financial Analyst",
        goal="Analyze financial data and news",
        llm=llm,
        verbose=True,
        backstory=f"You analyze stocks. Use Indian currency units. Today is {Today}",
        allow_delegation=False,
    )

    fin_expert = Agent(
        role="Investment Advisor",
        goal="Make investment recommendations",
        llm=llm,
        verbose=True,
        tools=[get_curr_stock_price],
        max_iter=2,
        backstory=f"You provide investment advice. Today is {Today}",
        allow_delegation=False,
    )

    get_company_financials = Task(
        description=f"Get financial data for {stock_symbol} using the tools available.",
        expected_output="Financial data with key metrics",
        agent=data_explorer,
    )

    get_company_news = Task(
        description=f"Search for latest news about {stock_symbol}",
        expected_output="Summary of recent news",
        agent=news_info_explorer,
    )

    analyse = Task(
        description="Analyze the financial data and news provided. Write a comprehensive report.",
        expected_output="Detailed financial analysis in markdown format",
        agent=analyst,
        context=[get_company_financials, get_company_news],
        output_file="Analysis.md",
    )

    advise = Task(
        description="Based on the analysis, recommend Buy, Hold, or Sell with clear reasons.",
        expected_output="Investment recommendation with reasoning in markdown",
        agent=fin_expert,
        context=[analyse],
        output_file="Recommendation.md",
    )

    crew = Crew(
        agents=[data_explorer, news_info_explorer, analyst, fin_expert],
        tasks=[get_company_financials, get_company_news, analyse, advise],
        verbose=True,
        process=Process.sequential,
    )

    result = crew.kickoff({"stock": stock_symbol})

    try:
        with open("Analysis.md", "r", encoding="utf-8") as f:
            analysis_md = f.read()
        with open("Recommendation.md", "r", encoding="utf-8") as f:
            recommendation_md = f.read()
        analysis_html = markdown2.markdown(analysis_md)
        recommendation_html = markdown2.markdown(recommendation_md)
        return analysis_html, recommendation_html
    except Exception as e:
        return f"<p>Error: {e}</p>", ""
