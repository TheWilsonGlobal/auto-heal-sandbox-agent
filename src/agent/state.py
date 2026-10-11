"""State representation for the Autonomous Code-Healing Agent."""

from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """LangGraph state schema for tracking code healing lifecycle."""

    task_description: str
    source_code: str
    test_code: str
    error_trace: Optional[str]
    fixed_code: Optional[str]
    iterations: int
    max_retries: int
    is_solved: bool

    # Detailed iteration tracking history for comprehensive logging & export
    iteration_history: Optional[list]

    # Ephemeral node execution transfer fields
    _exec_exit_code: Optional[int]
    _exec_stdout: Optional[str]
    _exec_stderr: Optional[str]
    _last_raw_llm_response: Optional[str]


