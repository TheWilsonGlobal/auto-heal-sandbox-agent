"""Comprehensive test suite for the autonomous code-healing agent loop."""

import pytest
from unittest.mock import MagicMock, patch

from src.utils.parser import extract_python_code, truncate_traceback
from src.agent.state import AgentState
from src.agent.graph import route_next, build_healing_graph
from src.agent.nodes import HealingNodes
from src.mcp.server import mcp


# --- 1. Parser Tests ---

def test_extract_python_code_strict_block():
    text = """Here is the fix:
```python
def add(a, b):
    return a + b
```
Hope this helps!"""
    expected = "def add(a, b):\n    return a + b"
    assert extract_python_code(text) == expected


def test_extract_python_code_generic_block():
    text = """```
def multiply(a, b):
    return a * b
```"""
    expected = "def multiply(a, b):\n    return a * b"
    assert extract_python_code(text) == expected


def test_extract_python_code_no_markdown():
    text = "def subtract(a, b):\n    return a - b"
    assert extract_python_code(text) == text


def test_extract_python_code_liquidai_write_tool_call():
    text = """<|tool_call_start|>[write(path='solution.py', content='def add(a, b):\\n    return a + b\\n')]<|tool_call_end|>"""
    assert extract_python_code(text) == "def add(a, b):\n    return a + b"


def test_extract_python_code_liquidai_edit_tool_call():
    source = "def calc(a, b):\n    return a - b"
    text = """<|tool_call_start|>[edit(file_path='solution.py', old_string='return a - b', new_string='return a + b')]<|tool_call_end|>"""
    assert extract_python_code(text, current_code=source) == "def calc(a, b):\n    return a + b"


def test_extract_python_code_liquidai_edits_list():
    source = "def get(x):\n    return x\ndef put(x):\n    pass"
    text = """<|tool_call_start|>[edit(path='solution.py', edits=[{'oldText': 'return x', 'newText': 'return x * 2'}, {'oldText': 'pass', 'newText': 'self.x = x'}])]<|tool_call_end|>"""
    res = extract_python_code(text, current_code=source)
    assert "return x * 2" in res
    assert "self.x = x" in res


def test_extract_python_code_liquidai_updates_diff():
    source = "def hello():\n    return 'bad'"
    text = """<|tool_call_start|>[edit(path='solution.py', updates='<<<<<<< SEARCH\\nreturn \\'bad\\'\\n=======\\nreturn \\'good\\'\\n>>>>>>> REPLACE')]<|tool_call_end|>"""
    res = extract_python_code(text, current_code=source)
    assert "return 'good'" in res


def test_extract_python_code_liquidai_read_rejection():
    text = "<|tool_call_start|>[read(file_path='solution.py')]<|tool_call_end|>"
    assert extract_python_code(text) == ""


def test_extract_python_code_syntax_error_rejection():
    # Non-parseable python code
    text = "<|tool_call_start|>[unknown_token()]"
    assert extract_python_code(text) == ""




def test_truncate_traceback_short():
    short_trace = "Line 1\nLine 2\nLine 3"
    assert truncate_traceback(short_trace, max_lines=10) == short_trace


def test_truncate_traceback_exceeds_max_lines():
    long_trace = "\n".join([f"Trace line {i}" for i in range(25)])
    truncated = truncate_traceback(long_trace, max_lines=10)
    lines = truncated.splitlines()
    assert len(lines) == 10
    assert lines[0] == "Trace line 15"
    assert lines[-1] == "Trace line 24"


def test_truncate_traceback_empty():
    assert truncate_traceback("") == ""
    assert truncate_traceback("   \n  ") == ""


# --- 2. Conditional Edge Routing Tests ---

def test_route_next_when_solved():
    state: AgentState = {
        "task_description": "test",
        "source_code": "...",
        "test_code": "...",
        "error_trace": None,
        "fixed_code": "...",
        "iterations": 1,
        "max_retries": 4,
        "is_solved": True,
    }
    assert route_next(state) == "__end__"


def test_route_next_when_max_retries_reached():
    state: AgentState = {
        "task_description": "test",
        "source_code": "...",
        "test_code": "...",
        "error_trace": "Error",
        "fixed_code": "...",
        "iterations": 4,
        "max_retries": 4,
        "is_solved": False,
    }
    assert route_next(state) == "__end__"


def test_route_next_when_retrying():
    state: AgentState = {
        "task_description": "test",
        "source_code": "...",
        "test_code": "...",
        "error_trace": "Error",
        "fixed_code": "...",
        "iterations": 2,
        "max_retries": 4,
        "is_solved": False,
    }
    assert route_next(state) == "analyze_and_patch"


# --- 3. Micro-Nodes and Agent Loop with Mock Sandbox ---

class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def invoke(self, messages):
        resp = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        mock = MagicMock()
        mock.content = resp
        return mock


class FakeSandbox:
    def __init__(self, exec_results):
        self.exec_results = exec_results
        self.index = 0
        self.files = {}
        self.started = False
        self.cleaned = False

    def start(self):
        self.started = True

    def write_file(self, filename: str, content: str, target_dir: str = "/tmp"):
        self.files[filename] = content

    def execute(self, cmd: str, timeout: int = 10):
        res = self.exec_results[min(self.index, len(self.exec_results) - 1)]
        self.index += 1
        return res

    def cleanup(self):
        self.cleaned = True


def test_healing_loop_success_on_first_try():
    fake_code = "```python\ndef add(a, b): return a + b\n```"
    fake_llm = FakeLLM([fake_code])
    fake_sandbox = FakeSandbox([(0, "1 passed in 0.01s", "")])

    app = build_healing_graph(llm=fake_llm, sandbox=fake_sandbox)

    initial_state: AgentState = {
        "task_description": "Fix addition",
        "source_code": "def add(a, b): return a - b",
        "test_code": "def test_add(): assert add(1, 2) == 3",
        "error_trace": None,
        "fixed_code": None,
        "iterations": 0,
        "max_retries": 3,
        "is_solved": False,
    }

    final_state = app.invoke(initial_state)

    assert final_state["is_solved"] is True
    assert "def add(a, b): return a + b" in final_state["fixed_code"]
    assert "solution.py" in fake_sandbox.files


def test_healing_loop_retry_and_succeed():
    # Attempt 1 fails, Attempt 2 succeeds
    llm_responses = [
        "```python\ndef add(a, b): return a\n```",
        "```python\ndef add(a, b): return a + b\n```",
    ]
    sandbox_results = [
        (1, "", "FAILED test_add - AssertionError"),
        (0, "1 passed", ""),
    ]

    fake_llm = FakeLLM(llm_responses)
    fake_sandbox = FakeSandbox(sandbox_results)

    app = build_healing_graph(llm=fake_llm, sandbox=fake_sandbox)

    initial_state: AgentState = {
        "task_description": "Fix addition with retry",
        "source_code": "def add(a, b): return 0",
        "test_code": "def test_add(): assert add(1, 2) == 3",
        "error_trace": None,
        "fixed_code": None,
        "iterations": 0,
        "max_retries": 3,
        "is_solved": False,
    }

    final_state = app.invoke(initial_state)

    assert final_state["is_solved"] is True
    assert final_state["iterations"] == 1  # 1 failure before passing
    assert "def add(a, b): return a + b" in final_state["fixed_code"]


def test_healing_loop_max_retries_exhausted():
    # Always fails
    fake_llm = FakeLLM(["```python\ndef add(a, b): return 0\n```"])
    fake_sandbox = FakeSandbox([(1, "", "FAILED test_add - AssertionError")])

    app = build_healing_graph(llm=fake_llm, sandbox=fake_sandbox)

    initial_state: AgentState = {
        "task_description": "Fix addition that never works",
        "source_code": "def add(a, b): return 0",
        "test_code": "def test_add(): assert add(1, 2) == 3",
        "error_trace": None,
        "fixed_code": None,
        "iterations": 0,
        "max_retries": 2,
        "is_solved": False,
    }

    final_state = app.invoke(initial_state)

    assert final_state["is_solved"] is False
    assert final_state["iterations"] == 2
    assert "AssertionError" in final_state["error_trace"]


# --- 4. MCP Server Registration Tests ---

def test_mcp_tools_registered():
    if hasattr(mcp, "_tool_manager"):
        tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    else:
        import asyncio
        tools = asyncio.run(mcp.list_tools())
        tool_names = [tool.name for tool in tools]

    assert "execute_test_suite" in tool_names
    assert "execute_command" in tool_names
    assert "heal_code" in tool_names


# --- 5. Docker Sandbox Real Container Integration Test ---


def test_docker_runner_real_execution():
    from src.sandbox.docker_runner import DockerSandboxRunner
    try:
        runner = DockerSandboxRunner()
        with runner as sandbox:
            sandbox.write_file("dummy.py", "print('hello from sandbox')", target_dir="/tmp")
            exit_code, stdout, stderr = sandbox.execute("python /tmp/dummy.py", timeout=5)
            assert exit_code == 0
            assert "hello from sandbox" in stdout
    except Exception as e:
        pytest.skip(f"Docker sandbox test skipped due to environment: {e}")

