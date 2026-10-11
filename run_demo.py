"""Demo runner executing the LangGraph autonomous code-healing loop."""

import sys
import os
import argparse
import logging
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import settings
from src.agent.graph import create_healing_agent
from src.agent.state import AgentState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("auto-heal-demo")


class MockLFM25LLM:
    """Mock LLM simulating LiquidAI LFM 2.5 2.6B responses for offline demo testing."""

    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        logger.info("[Mock LLM] Generating simulated patch for iteration %d...", self.call_count)

        # Corrected implementation of Calculator
        fixed_code = '''```python
from typing import List, Union

class Calculator:
    """A mathematical operations calculator with all bugs healed."""

    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def power(self, base: float, exponent: float) -> float:
        return base ** exponent

    def average(self, numbers: List[Union[int, float]]) -> float:
        if not numbers:
            return 0.0
        return sum(numbers) / len(numbers)

    def factorial(self, n: int) -> int:
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        return n * self.factorial(n - 1)
```'''
        class FakeResponse:
            content = fixed_code
        return FakeResponse()


def check_local_llm_running(endpoint: str = settings.LLM_BASE_URL) -> bool:
    """Check if the local llama.cpp endpoint is responding."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{endpoint}/models", headers={"User-Agent": "python"})
        with urllib.request.urlopen(req, timeout=1.5):
            return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Autonomous Code-Healing Agent Demo")
    parser.add_argument("--mock", action="store_true", help="Force using mock LLM for testing")
    parser.add_argument("--retries", type=int, default=settings.MAX_RETRIES, help="Max retries")
    args = parser.parse_args()

    print("=" * 70)
    print("   AUTONOMOUS CODE-HEALING AGENT (Docker Sandbox + LFM 2.5 2.6B)")
    print("=" * 70)

    # 1. Load sample buggy files
    sample_dir = REPO_ROOT / "sample_bugs"
    buggy_file = sample_dir / "buggy_calculator.py"
    test_file = sample_dir / "test_calculator.py"

    if not buggy_file.exists() or not test_file.exists():
        logger.error("Sample bug files not found in %s", sample_dir)
        sys.exit(1)

    source_code = buggy_file.read_text(encoding="utf-8")
    test_code = test_file.read_text(encoding="utf-8")

    print(f"\n[1] Loaded Buggy Code ({len(source_code)} bytes):")
    print("-" * 50)
    print("\n".join(source_code.splitlines()[:15]) + "\n... [truncated]")
    print("-" * 50)

    print(f"\n[2] Loaded Test Suite ({len(test_code)} bytes):")
    print("-" * 50)
    print("\n".join(test_code.splitlines()[:15]) + "\n... [truncated]")
    print("-" * 50)

    # 2. Configure LLM
    custom_llm = None
    if args.mock:
        logger.info("Using Mock LLM as explicitly requested via --mock.")
        custom_llm = MockLFM25LLM()
    else:
        is_live = check_local_llm_running()
        if is_live:
            logger.info("Connected to local LLM at %s (Model: %s)", settings.LLM_BASE_URL, settings.LLM_MODEL_NAME)
        else:
            logger.warning(
                "Local LLM not detected at %s. Falling back to Mock LLM for demo.",
                settings.LLM_BASE_URL,
            )
            custom_llm = MockLFM25LLM()

    # 3. Build Agent
    print("\n[3] Compiling LangGraph Code-Healing Workflow...")
    agent = create_healing_agent(llm=custom_llm)

    initial_state: AgentState = {
        "task_description": "Fix ZeroDivisionError, off-by-one average calculation, and negative factorial handling in Calculator.",
        "source_code": source_code,
        "test_code": test_code,
        "error_trace": None,
        "fixed_code": None,
        "iterations": 0,
        "max_retries": args.retries,
        "is_solved": False,
    }

    # 4. Execute Workflow
    print("\n[4] Running Cyclic Execution Loop (Plan -> Patch -> Docker Sandbox -> Evaluate)...")
    final_state = agent.invoke(initial_state)

    print("\n" + "=" * 70)
    print("                     EXECUTION RESULTS")
    print("=" * 70)
    print(f"Solved Status:      {final_state.get('is_solved')}")
    print(f"Iterations:         {final_state.get('iterations')}")

    if final_state.get("is_solved"):
        print("\n[+] Final Healed Code:")
        print("-" * 50)
        print(final_state.get("fixed_code"))
        print("-" * 50)
        print("\nAll unit tests passed inside isolated Docker sandbox!")
    else:
        print("\n[-] Healing loop failed to satisfy unit tests within retry budget.")
        print(f"Last Error Trace:\n{final_state.get('error_trace')}")


if __name__ == "__main__":
    main()
