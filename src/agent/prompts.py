"""Strict, small-footprint prompts optimized for LiquidAI LFM 2.5 (2.6B)."""

import re
from constants import CODE_PATCH_SYSTEM_PROMPT, CODE_PATCH_USER_PROMPT


def extract_failure_diagnostics(trace: str) -> list[str]:
    """Extract actionable diagnostic hints from pytest failure traceback."""
    diagnostics = []
    if not trace:
        return diagnostics

    # Detect KeyError (e.g., del dict[key] when key does not exist or key was never in dict)
    key_error_match = re.search(r"KeyError:\s*['\"]?(\w+)['\"]?", trace)
    if key_error_match:
        k = key_error_match.group(1)
        diagnostics.append(
            f"- CRITICAL KEYERROR ON `{k}`: A `KeyError: '{k}'` occurred (likely during `del self.cache[key]`). "
            f"NEVER delete `key` without checking `if key in self.cache` first! "
            f"If `key not in self.cache`: return None immediately (do NOT delete anything). "
            f"Only delete `key` if it is actually present in the dictionary (e.g. `if key in self.cache: del self.cache[key]`)."
        )

    # Detect 'assert key not in container' failures
    not_in_matches = re.findall(r"assert\s+['\"]?(\w+)['\"]?\s+not\s+in\s+([\w\.]+)", trace)
    for key, container in not_in_matches:
        diagnostics.append(
            f"- KEY CLEANUP FAILURE: `{key}` was still found in `{container}`! "
            f"When an item expires, check `if key in self.cache: del self.cache[key]` and clean up from expiry and order."
        )

    # Detect 'assert expr is None' failures
    is_none_matches = re.findall(r"assert\s+([^=\n]+?)\s+is\s+None", trace)
    for expr in is_none_matches:
        expr_clean = expr.strip()
        if not expr_clean.startswith("short test"):
            diagnostics.append(
                f"- NULL/EXPIRY RETURN FAILURE: `{expr_clean}` should have returned None."
            )

    return diagnostics


def format_error_section(error_trace: str | None) -> str:
    """Format the pruned error traceback section with diagnostic hints for the LLM prompt."""
    if not error_trace or not error_trace.strip():
        return "ERROR TRACEBACK:\nNo previous execution traceback. Run first iteration."

    diagnostics = extract_failure_diagnostics(error_trace)
    diag_text = ""
    if diagnostics:
        diag_text = (
            "\nDIAGNOSTIC HINTS (FROM TEST FAILURES):\n"
            + "\n".join(diagnostics)
            + "\n"
        )

    return (
        "FAILING TEST RESULTS & TRACEBACK:\n"
        "```text\n"
        f"{error_trace.strip()}\n"
        "```\n"
        f"{diag_text}"
        "Analyze the assertion failures above carefully. In particular, ensure all expired or evicted items "
        "are completely deleted/removed from all dictionaries and access lists so that assertions pass."
    )

