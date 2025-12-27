# AgentForge

> **Enterprise AI Platform** - Strategic Swarms, Production RAG, and Evaluation Pipelines.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange.svg)](https://langchain-ai.github.io/langgraph/)

---

## 🚀 Overview

**AgentForge** is a production-grade AI platform designed to demonstrate **Senior AI Engineer** capabilities. It moves beyond simple scripts to offer a robust, containerized architecture featuring decoupled microservices, autonomous agent swarms, and rigorous evaluation pipelines.

### Key Capabilities
- **🐝 Strategic Swarms**: Parallel agent orchestration (Planner → Analysts → Aggregator) using **LangGraph**.
- **🧪 Evaluation Lab**: Automated RAG accuracy testing using **LLM-as-a-Judge** (Faithfulness/Correctness metrics).
- **🛠️ Codebase Analyst**: Autonomous agents that can read and analyze the local repository structure.
- **🏗️ Decoupled Architecture**: Separate **FastAPI Backend** and **Streamlit Frontend** services.
- **🐳 Cloud-Ready**: Fully Dockerized with `docker-compose` orchestration and CI/CD workflows.

---

## 🏗️ Architecture

AgentForge uses a hybrid microservices pattern:

```mermaid
graph TD
    Client[User Browser] -->|HTTP| UI[Streamlit Frontend (:8501)]
    UI -->|REST API| API[FastAPI Backend (:8000)]
    API -->|Orchestration| Brain[LangGraph Engine]
    Brain -->|Parallel| Agents[Gemini/Claude Agents]
    Brain -->|Retrieve| VectorDB[ChromaDB]
    API -->|Track| Logs[Observability Middleware]
```

---

## 🛠️ Quick Start (Docker - Recommended)

The easiest way to run the full platform is via Docker Compose.

```bash
# 1. Clone the repo
git clone https://github.com/ChunkyTortoise/agentforge.git
cd agentforge

# 2. Configure Environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (Required)

# 3. Launch Services
docker compose up --build
```

Access the application:
- **Frontend**: [http://localhost:8501](http://localhost:8501)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💻 Local Development

If you prefer running without Docker:

```bash
# 1. Install Dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start Backend (Terminal 1)
uvicorn api.main:app --reload --port 8000

# 3. Start Frontend (Terminal 2)
streamlit run app.py
```

---

## 🧩 Modules

| Module | Features | Technical Stack |
|--------|----------|----------------|
| **Agent Hub** | Parallel Swarms, Code Analysis | LangGraph, FastAPI, AST |
| **Evaluation Lab** | RAG scoring, Latency tracking | Ragas-style metrics, Cosine Sim |
| **RAG Assistant** | Document Q&A with citations | LangChain, ChromaDB |
| **Smart Analyst** | AI-driven Data Exploration | PandasAI, Plotly |

---

## 🛡️ Quality Assurance

We employ a "Shift-Left" testing strategy:

- **Unit Tests**: `tests/unit/` (Mocked external dependencies)
- **Integration Tests**: `tests/test_api.py` (FastAPI endpoint validation)
- **CI/CD**: GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push.

---

## 👨‍💻 Author

**Cayman Roden** - Senior AI Engineer
*Demonstrating the bridge between experimental AI and production engineering.*