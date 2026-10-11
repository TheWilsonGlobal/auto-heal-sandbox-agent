"""FastMCP server wrapping sandbox operations and agent healing."""

import json
import logging
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.config import settings
from src.sandbox.docker_runner import DockerSandboxRunner
from src.agent.graph import create_healing_agent

logger = logging.getLogger(__name__)

mcp = FastMCP(settings.MCP_SERVER_NAME)


@mcp.tool()
def execute_test_suite(
    source_code: str,
    test_code: str,
    timeout: int = 10,
) -> str:
    """Execute Python code against a pytest suite in the isolated Docker sandbox.

    Args:
        source_code: The Python solution code
        test_code: The companion pytest test suite
        timeout: Execution timeout in seconds (default 10s)

    Returns:
        JSON string containing exit_code, stdout, stderr, and passed status.
    """
    with DockerSandboxRunner() as sandbox:
        sandbox.write_file("solution.py", source_code, target_dir="/tmp")
        sandbox.write_file("test_suite.py", test_code, target_dir="/tmp")

        cmd = "python -m pytest -v /tmp/test_suite.py"
        exit_code, stdout, stderr = sandbox.execute(cmd, timeout=timeout)

        result: Dict[str, Any] = {
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
        }
        return json.dumps(result, indent=2)


@mcp.tool()
def execute_command(
    command: str,
    timeout: int = 10,
) -> str:
    """Execute an arbitrary bash command inside the isolated Docker sandbox.

    Args:
        command: The shell command to run inside /tmp
        timeout: Execution timeout in seconds (default 10s)

    Returns:
        JSON string containing exit_code, stdout, and stderr.
    """
    with DockerSandboxRunner() as sandbox:
        exit_code, stdout, stderr = sandbox.execute(command, timeout=timeout)
        result: Dict[str, Any] = {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }
        return json.dumps(result, indent=2)


@mcp.tool()
def heal_code(
    task_description: str,
    source_code: str,
    test_code: str,
    max_retries: int = 4,
) -> str:
    """Run the complete cyclic LangGraph healing agent loop on buggy code.

    Args:
        task_description: Summary of what the code is supposed to do
        source_code: The buggy Python implementation
        test_code: The pytest test suite defining correct behavior
        max_retries: Maximum number of repair iterations

    Returns:
        JSON string containing is_solved, iterations, fixed_code, and error_trace.
    """
    agent = create_healing_agent()
    initial_state = {
        "task_description": task_description,
        "source_code": source_code,
        "test_code": test_code,
        "error_trace": None,
        "fixed_code": None,
        "iterations": 0,
        "max_retries": max_retries,
        "is_solved": False,
    }

    final_state = agent.invoke(initial_state)
    output = {
        "is_solved": final_state.get("is_solved", False),
        "iterations": final_state.get("iterations", 0),
        "fixed_code": final_state.get("fixed_code"),
        "error_trace": final_state.get("error_trace"),
    }
    return json.dumps(output, indent=2)


def create_mcp_server() -> FastMCP:
    """Return configured FastMCP server instance."""
    return mcp


if __name__ == "__main__":
    mcp.run()
