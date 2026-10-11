"""Strict, small-footprint prompts optimized for LiquidAI LFM 2.5 (2.6B)."""

CODE_PATCH_SYSTEM_PROMPT = """You are an expert autonomous Python bug-fixing engine.
Your goal is to inspect the buggy Python source code, review the failing pytest suite and error traceback, and produce the complete corrected Python code.

CRITICAL INSTRUCTIONS:
1. Provide the complete corrected Python code that will make all tests pass.
2. Output the code inside a single ```python ... ``` block or via write(path='solution.py', content='...').
3. Do NOT output conversational text, explanations, or questions.
4. Preserve existing functions and classes, fixing all bugs preventing tests from passing.
5. Pay close attention to assertion failures, cleanup requirements, and edge cases in the test suite.
"""

CODE_PATCH_USER_PROMPT = """TASK DESCRIPTION:
{task_description}

CURRENT CODE (solution.py):
```python
{source_code}
```

TEST SUITE (pytest):
```python
{test_code}
```

{error_section}

Write the complete corrected Python code for solution.py to pass all tests:
"""


def format_error_section(error_trace: str | None) -> str:
    """Format the pruned error traceback section for the LLM prompt."""
    if not error_trace or not error_trace.strip():
        return "ERROR TRACEBACK:\nNo previous execution traceback. Run first iteration."
    return (
        "FAILING TEST RESULTS & TRACEBACK:\n"
        "```text\n"
        f"{error_trace.strip()}\n"
        "```\n"
        "Analyze the assertion failures above. Fix all logic flaws so that every test in the suite passes completely."
    )
