"""LangGraph execution nodes for the autonomous healing loop."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.agent.state import AgentState
from src.agent.prompts import (
    CODE_PATCH_SYSTEM_PROMPT,
    CODE_PATCH_USER_PROMPT,
    format_error_section,
)
from src.sandbox.docker_runner import DockerSandboxRunner
from src.utils.parser import extract_python_code, truncate_traceback

logger = logging.getLogger(__name__)


def get_llm(custom_llm: Optional[Any] = None) -> Any:
    """Instantiate the OpenAI-compatible client for local LFM 2.5."""
    if custom_llm is not None:
        return custom_llm

    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
    )


class HealingNodes:
    """Encapsulates node functions with optional dependency injection for testing."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        sandbox: Optional[DockerSandboxRunner] = None,
    ):
        self.llm = get_llm(llm)
        self.sandbox = sandbox

    def analyze_and_patch(self, state: AgentState) -> Dict[str, Any]:
        """Analyze current failures and prompt LLM to generate corrected Python code."""
        current_iter = state.get("iterations", 0)
        max_retries = state.get("max_retries", settings.MAX_RETRIES)
        logger.info("--- [NODE: analyze_and_patch] Iteration %d/%d ---", current_iter, max_retries)

        current_code = state.get("fixed_code") or state.get("source_code", "")
        error_section = format_error_section(state.get("error_trace"))

        prompt_text = CODE_PATCH_USER_PROMPT.format(
            task_description=state.get("task_description", "Fix bugs to pass all tests"),
            source_code=current_code,
            test_code=state.get("test_code", ""),
            error_section=error_section,
        )

        messages = [
            SystemMessage(content=CODE_PATCH_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_response = getattr(response, "content", str(response))
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raw_response = f"```python\n{current_code}\n```"

        patched_code = extract_python_code(raw_response, current_code=current_code)
        recorded_llm_response = raw_response

        # Handle LiquidAI tool-call requests (e.g. read_file / edit_plan) with a tool-response turn
        if not patched_code.strip() and ("<|tool_call_start|>" in raw_response):
            logger.info("Model requested tool operation (%s). Providing file contents in turn 2...", raw_response[:60])
            try:
                followup_messages = list(messages)
                followup_messages.append(AIMessage(content=raw_response))
                followup_messages.append(
                    HumanMessage(
                        content=(
                            f"<|tool_response_start|>File content of solution.py:\n{current_code}<|tool_response_end|>\n\n"
                            f"{error_section}\n\n"
                            "Now write the complete corrected solution.py to pass all tests."
                        )
                    )
                )
                followup_resp = self.llm.invoke(followup_messages)
                raw_followup = getattr(followup_resp, "content", str(followup_resp))
                recorded_llm_response = f"Turn 1: {raw_response}\n\nTurn 2: {raw_followup}"
                patched_code = extract_python_code(raw_followup, current_code=current_code)
            except Exception as e:
                logger.error("Follow-up turn failed: %s", e)

        # Safety fallback to current_code if extraction was empty or invalid
        if not patched_code.strip():
            logger.warning("Empty or invalid code extracted from LLM. Falling back to current code.")
            patched_code = current_code

        logger.info("Generated code patch (%d bytes)", len(patched_code))
        return {
            "fixed_code": patched_code,
            "_last_raw_llm_response": recorded_llm_response,
        }

    def execute_in_sandbox(self, state: AgentState) -> Dict[str, Any]:
        """Write current fixed code and tests to the sandbox, then run pytest."""
        logger.info("--- [NODE: execute_in_sandbox] Running pytest in sandbox ---")
        code_to_test = state.get("fixed_code") or state.get("source_code", "")
        test_code = state.get("test_code", "")

        sandbox = self.sandbox
        managed_sandbox = False

        if sandbox is None:
            sandbox = DockerSandboxRunner()
            sandbox.start()
            managed_sandbox = True

        try:
            # Write solution and test suite to sandbox /tmp
            sandbox.write_file("solution.py", code_to_test, target_dir="/tmp")
            sandbox.write_file("test_suite.py", test_code, target_dir="/tmp")

            # Execute pytest inside sandbox with timeout
            cmd = "python -m pytest -v /tmp/test_suite.py"
            exit_code, stdout, stderr = sandbox.execute(
                cmd,
                timeout=settings.SANDBOX_TIMEOUT_SECONDS,
            )

            logger.info("Sandbox execution finished with exit code: %d", exit_code)
            return {
                "_exec_exit_code": exit_code,
                "_exec_stdout": stdout,
                "_exec_stderr": stderr,
            }
        finally:
            if managed_sandbox:
                sandbox.cleanup()

    def evaluate_result(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate sandbox execution exit code, record detailed history, and update state."""
        logger.info("--- [NODE: evaluate_result] Evaluating test results ---")
        exit_code = state.get("_exec_exit_code", 1)
        stdout = state.get("_exec_stdout", "")
        stderr = state.get("_exec_stderr", "")
        current_iterations = state.get("iterations", 0)

        # Prune traceback if failed
        combined_output = stderr.strip() if stderr.strip() else stdout.strip()
        pruned_trace = truncate_traceback(
            combined_output,
            max_lines=settings.TRACEBACK_MAX_LINES,
        ) if exit_code != 0 else None

        # Build detailed iteration history entry
        history_entry = {
            "iteration": current_iterations + 1,
            "timestamp": datetime.now().isoformat(),
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "fixed_code": state.get("fixed_code"),
            "stdout": stdout,
            "stderr": stderr,
            "pruned_error_trace": pruned_trace,
            "raw_llm_response": state.get("_last_raw_llm_response"),
        }

        history: List[Dict[str, Any]] = list(state.get("iteration_history") or [])
        history.append(history_entry)

        if exit_code == 0:
            logger.info("SUCCESS: All tests passed on iteration %d!", current_iterations)
            return {
                "is_solved": True,
                "error_trace": None,
                "iteration_history": history,
            }

        # Failed execution
        logger.warning("Test suite failed (exit code %d).", exit_code)
        return {
            "is_solved": False,
            "iterations": current_iterations + 1,
            "error_trace": pruned_trace,
            "iteration_history": history,
        }
