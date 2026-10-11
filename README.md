# Auto-Heal Sandbox Agent 🛡️

> **Autonomous Code-Healing Agent with isolated Docker Sandbox, LangGraph cyclic state-machine, and local LiquidAI LFM 2.5 (2.6B) inference.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/sandbox-Docker_Isolated-2496ED.svg)](https://www.docker.com/)
[![Local LLM](https://img.shields.io/badge/model-LiquidAI_LFM_2.5_2.6B-green.svg)](https://www.liquid.ai/)
[![MCP](https://img.shields.io/badge/protocol-FastMCP-purple.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

### 🏷️ GitHub Repository Topics
`ai-agent` • `autonomous-agent` • `code-healing` • `self-healing-code` • `langgraph` • `docker-sandbox` • `liquidai` • `lfm` • `local-llm` • `llama-cpp` • `model-context-protocol` • `mcp` • `pytest` • `streamlit` • `safe-eval` • `python`

---


## 1. Architecture Overview

```
                      +-----------------------------+
                      |   Buggy Code + Unit Tests   |
                      +--------------+--------------+
                                     |
                                     v
                        +--------------------------+
                  +---->|    analyze_and_patch     |<----+
                  |     |  (LFM 2.5 2.6B Inference)|     |
                  |     +------------+-------------+     |
                  |                  |                   |
                  |                  v                   |
                  |     +--------------------------+     | (Retry Loop)
                  |     |    execute_in_sandbox    |     |
                  |     | (Docker SDK / Pytest run)|     |
                  |     +------------+-------------+     |
                  |                  |                   |
                  |                  v                   |
                  |     +--------------------------+     |
                  |     |     evaluate_result      |     |
                  |     +------------+-------------+     |
                  |                  |                   |
                  |       Exit != 0? | Exit == 0?        |
                  +------------------+                   v
                 (if iters < max_retries)             [ END ]
                                                 (Healed Solution)
```

### Core Architecture Guardrails for 2.6B Models
- **Micro-Nodes with Strict Markdown Block Parsing**: Smaller LLMs (e.g., LiquidAI LFM 2.5 2.6B) can struggle with complex zero-shot multi-parameter JSON function calling. We bypass fragile function-calling schemas by using dedicated micro-nodes paired with robust regex block extraction (`r"```python\s*(.*?)\s*```"`).
- **Traceback Pruning**: Automatically extracts the core final 10 lines of exception traces (`truncate_traceback`), preserving limited context windows and preventing catastrophic distraction.
- **Strict Docker Sandboxing**:
  - Memory limit: `512MB`
  - CPU allocation: `1 core` (`nano_cpus=1_000_000_000`)
  - Network isolation: `network_mode="none"` (zero internet egress/ingress)
  - Security privilege: `user="nobody"`
  - Ephemeral file exchange: In-memory `tarfile` archives via `put_archive`
  - Timeout enforcement: 10s per command execution

---

## 2. Directory Structure

```text
auto-heal-sandbox-agent/
├── pyproject.toml              # Dependencies and build configuration
├── README.md                   # System documentation and operational guide
├── docker/
│   └── sandbox.Dockerfile      # Hardened unprivileged sandbox container image
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic BaseSettings (LLM and Docker configs)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py            # TypedDict AgentState schema
│   │   ├── prompts.py          # Small-model prompt templates
│   │   ├── nodes.py            # Micro-nodes (analyze_and_patch, execute, evaluate)
│   │   └── graph.py            # LangGraph cyclic state graph with conditional router
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseSandbox abstract interface
│   │   └── docker_runner.py    # Docker SDK isolated container runner
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py           # FastMCP server exposing sandbox tools
│   └── utils/
│       ├── __init__.py
│       └── parser.py           # Robust code extractors & traceback pruner
├── sample_bugs/
│   ├── buggy_calculator.py     # Buggy calculator with edge-case flaws
│   └── test_calculator.py      # Companion pytest verification suite
├── app.py                      # Streamlit interactive Web UI
├── run_demo.py                 # CLI demo runner with real & mock LLM support
└── tests/
    ├── __init__.py
    └── test_agent_loop.py      # Comprehensive unit tests
```

---

## 3. Setup & Installation

### Prerequisites
1. Python 3.10+ (or 3.11+)
2. Docker Desktop running locally
3. (Optional) Local llama.cpp server running LiquidAI LFM 2.5 2.6B at `http://localhost:5813/v1`

### Step 1: Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### Step 2: Build the Sandbox Docker Image
```bash
docker build -t auto-heal-sandbox:latest -f docker/sandbox.Dockerfile .
```

---

## 4. Running the Healing Agent

### Option A: CLI Demo
Run the command-line demo with live LLM (or automatic fallback to simulator if local endpoint is offline):
```bash
python run_demo.py
```

Force simulation mode:
```bash
python run_demo.py --mock
```

### Option B: Interactive Streamlit Web UI
Launch the browser UI:
```bash
streamlit run app.py
```
Features:
- Side-by-side editable text areas for buggy code and unit tests
- Real-time checkpoint tracker streaming LangGraph node transitions
- One-click load of sample bug reproduction cases
- Syntax-highlighted healed code output

### Option C: FastMCP Server
Run the Model Context Protocol server over stdio:
```bash
python src/mcp/server.py
```
Available MCP tools:
- `execute_test_suite(source_code, test_code, timeout=10)`
- `execute_command(command, timeout=10)`
- `heal_code(task_description, source_code, test_code, max_retries=4)`

---

## 5. Running Tests

Run the complete test suite:
```bash
pytest -v
```
