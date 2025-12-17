# AgentForge

> **AI-Powered Intelligence Platform** - RAG, Multi-Agent Systems, and Data Analytics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is AgentForge?

AgentForge is an AI-powered platform demonstrating production-grade implementations of:

- **🔍 RAG (Retrieval-Augmented Generation)** - Document Q&A with citations
- **🤖 Multi-Agent Systems** - Autonomous AI agents working together
- **📊 AI-Enhanced Data Analytics** - Natural language data exploration
- **⚡ Prompt Engineering** - Optimized prompt templates and testing
- **🔄 Multi-Model Support** - Gemini and Claude integration

**Built to showcase expertise in**: AI/ML Engineering, GenAI, LangChain, RAG, Agentic AI, and Data Analytics.

---

## Modules

| Module | Description | Certifications Showcased |
|--------|-------------|-------------------------|
| **RAG Assistant** | Upload documents, ask questions, get cited answers | IBM RAG/Agentic AI, Duke LLMOps |
| **Agent Hub** | Multi-agent research and analysis workflows | Google Cloud GenAI Agents, IBM Agentic AI |
| **Smart Analyst** | Natural language data exploration with AI | Google Data Analytics, Microsoft GenAI for Data |
| **Prompt Lab** | Prompt engineering and optimization tools | Vanderbilt Prompt Engineering |
| **Model Arena** | Compare LLM responses side-by-side | DeepLearning.AI, Duke LLMOps |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | Streamlit |
| **LLMs** | Google Gemini, Anthropic Claude |
| **AI/ML** | LangChain, LangGraph, ChromaDB |
| **Embeddings** | sentence-transformers, OpenAI |
| **Data** | Pandas, NumPy, Plotly |
| **Vector DB** | ChromaDB, FAISS |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ChunkyTortoise/agentforge.git
cd agentforge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
streamlit run app.py
```

---

## API Keys Required

| Provider | Purpose | Get Key |
|----------|---------|---------|
| Google Gemini | LLM (free tier) | [ai.google.dev](https://ai.google.dev/) |
| Anthropic Claude | LLM (paid) | [console.anthropic.com](https://console.anthropic.com/) |

---

## Project Structure

```
AgentForge/
├── app.py                 # Main Streamlit application
├── modules/               # Feature modules
│   ├── rag_assistant.py   # RAG document Q&A
│   ├── agent_hub.py       # Multi-agent orchestration
│   ├── smart_analyst.py   # AI data analytics
│   ├── prompt_lab.py      # Prompt engineering
│   └── model_arena.py     # LLM comparison
├── core/                  # Core AI components
│   ├── llm_client.py      # Unified LLM interface
│   ├── rag_engine.py      # RAG retrieval logic
│   ├── embeddings.py      # Vector embeddings
│   └── agents.py          # Agent definitions
├── utils/                 # Utilities
│   ├── document_loader.py # PDF/text processing
│   └── logger.py          # Logging
└── tests/                 # Test suite
```

---

## Certifications Behind This Project

This project demonstrates skills from **1,768+ hours** of professional certifications:

**AI/ML Engineering**:
- DeepLearning.AI Deep Learning Specialization (120h)
- IBM RAG and Agentic AI Professional Certificate
- IBM GenAI Engineering with PyTorch & LangChain
- Duke University LLMOps Specialization
- Google Cloud Generative AI Leader

**Data Analytics**:
- Google Data Analytics Professional Certificate (181h)
- Google Advanced Data Analytics Certificate
- IBM Business Intelligence Analyst (141h)
- Microsoft GenAI for Data Analysis

---

## Author

**Cayman Roden** - AI/ML Engineer

- GitHub: [@ChunkyTortoise](https://github.com/ChunkyTortoise)
- LinkedIn: [linkedin.com/in/caymanroden](https://linkedin.com/in/caymanroden)

---

## License

MIT License - see [LICENSE](LICENSE) for details.
