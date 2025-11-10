# Real-Time GenAI Investment Advisory System

## Overview
This project introduces a next-generation real-time investment advisory system that integrates **agentic AI orchestration**, **generative AI models**, and **live financial market data streams**. The system leverages **CrewAI** for multi-agent coordination and **OpenRouter** to access state-of-the-art LLMs. Market pricing and financial statements are sourced using **yfinance**, while global news sentiment is extracted using **DuckDuckGo Search**.  
The system provides **personalized investment recommendations**, **sentiment-driven insights**, and an **interactive chatbot** interface capable of answering context-aware financial questions.

## Features
- **CrewAI-based modular multi-agent architecture**
  - News research agent  
  - Financial data extraction agent  
  - Analysis agent  
  - Investment advisory agent  

- **Real-time streaming compatibility** (Apache Kafka / Flink ready)

- **Market data acquisition**
  - Live price and company metadata via **yfinance**
  - Financial news aggregation using **DuckDuckGo**

- **NLP Processing**
  - LLM-powered summarization via **OpenRouter**
  - Sentiment and reasoning-based investment signals

- **Context-aware chatbot**
  - Dynamically re-trained on each stock's analysis
  - Supports follow-up investment queries

- **Auditability & Transparency**
  - Model reasoning and decision context preserved

- **Low-latency real-time inference pipeline**
  - Insight delivery within seconds of data retrieval


## Installation
### 1. Clone the repository
```bash
git clone https://github.com/nithishvarsrinivasan/AgenticAI-Investment-Adivisor.git
cd AgenticAI-Investment-Adivisor
