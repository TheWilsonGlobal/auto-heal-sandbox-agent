"""Parser utilities for robust code extraction, tool-call interpretation, and traceback pruning."""

import ast
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _normalize_newlines(text: str) -> str:
    """Normalize line endings to standard LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _apply_search_replace_patch(code: str, old_chunk: str, new_chunk: str) -> str:
    """Apply a replacement with multiple fallback strategies for whitespace/indentation."""
    code_norm = _normalize_newlines(code)
    old_norm = _normalize_newlines(old_chunk)
    new_norm = _normalize_newlines(new_chunk)

    # 1. Exact match
    if old_norm in code_norm:
        return code_norm.replace(old_norm, new_norm, 1)

    # 2. Stripped match
    old_stripped = old_norm.strip()
    new_stripped = new_norm.strip()
    if old_stripped and old_stripped in code_norm:
        return code_norm.replace(old_stripped, new_stripped, 1)

    # 3. Line-by-line whitespace-tolerant match
    old_lines = [line.strip() for line in old_norm.splitlines() if line.strip()]
    if old_lines:
        code_lines = code_norm.splitlines()
        for i in range(len(code_lines) - len(old_lines) + 1):
            window = [code_lines[i + j].strip() for j in range(len(old_lines))]
            if window == old_lines:
                # Replace the window with new_norm
                prefix = "\n".join(code_lines[:i])
                suffix = "\n".join(code_lines[i + len(old_lines):])
                parts = [p for p in (prefix, new_norm, suffix) if p]
                return "\n".join(parts)

    logger.warning("Could not find matching target block in source code for patch.")
    return code_norm


def _parse_diff_updates(updates_text: str, current_code: str) -> str:
    """Parse updates containing *** Begin Patch or <<<<<<< SEARCH blocks."""
    code = _normalize_newlines(current_code)

    # Strategy A: Aider / SWE-bench SEARCH / REPLACE blocks
    search_replace_pattern = re.compile(
        r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
        re.DOTALL,
    )
    matches = search_replace_pattern.findall(updates_text)
    if matches:
        for old_part, new_part in matches:
            code = _apply_search_replace_patch(code, old_part, new_part)
        return code

    # Strategy B: *** Begin Patch unified hunks
    hunk_pattern = re.compile(
        r"@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@\n(.*?)(?=\n@@|\n\*\*\*|$)",
        re.DOTALL,
    )
    hunks = hunk_pattern.findall(updates_text)
    if hunks:
        for hunk in hunks:
            old_lines = []
            new_lines = []
            for line in hunk.splitlines():
                if line.startswith("-"):
                    old_lines.append(line[1:])
                elif line.startswith("+"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])
                elif not line.startswith("+") and not line.startswith("-"):
                    old_lines.append(line)
                    new_lines.append(line)
            old_block = "\n".join(old_lines)
            new_block = "\n".join(new_lines)
            code = _apply_search_replace_patch(code, old_block, new_block)
        return code

    return code


def extract_python_code(text: str, current_code: str = "") -> str:
    """Extract Python source code from LLM response.

    Supports:
    1. Standard ```python ... ``` and ``` ... ``` markdown blocks.
    2. LiquidAI LFM 2.5 native tool-calling syntax:
       - <|tool_call_start|>[write(path=..., content='...')]<|tool_call_end|>
       - <|tool_call_start|>[edit(path=..., edits=[{'oldText': ..., 'newText': ...}])]
       - <|tool_call_start|>[edit(path=..., updates='...')]
       - <|tool_call_start|>[edit(path=..., old_string=..., new_string=...)]
    3. Rejection of non-code tool calls (e.g. read_file, edit_plan).
    4. Python AST syntax verification to prevent deploying syntax-broken files.
    """
    if not text:
        return ""

    candidate = ""

    # 1. Primary: strict ```python ... ``` block
    python_block_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    matches = python_block_pattern.findall(text)
    if matches:
        candidate = matches[-1].strip()

    # 2. Fallback: generic ``` ... ``` block
    if not candidate:
        generic_block_pattern = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)
        generic_matches = generic_block_pattern.findall(text)
        if generic_matches:
            candidate = generic_matches[-1].strip()

    # 3. Check for LiquidAI / tool-calling syntax: <|tool_call_start|>...<|tool_call_end|>
    if not candidate or "<|tool_call_start|>" in text:
        tool_matches = re.findall(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text, re.DOTALL)
        payloads = tool_matches if tool_matches else [text]

        for p in payloads:
            p_clean = p.strip()

            # Pattern A: write(..., content='...')
            write_match = re.search(
                r"write\s*\([^)]*?\bcontent\s*=\s*(['\"]{1,3})(.*?)\1\s*(?:,\s*\w+=[^)]*)*\)",
                p_clean,
                re.DOTALL,
            )
            if write_match:
                raw_content = write_match.group(2)
                try:
                    candidate = raw_content.encode("utf-8").decode("unicode_escape").strip()
                    break
                except Exception:
                    candidate = raw_content.strip()
                    break

            # Pattern B: AST evaluation of tool expression (handles edits list, write, etc.)
            try:
                tree = ast.parse(p_clean)
                expr = tree.body[0].value
                calls = expr.elts if isinstance(expr, ast.List) else [expr]
                for call in calls:
                    if isinstance(call, ast.Call):
                        fn_name = getattr(call.func, "id", "")

                        # Handle write(content=...)
                        if fn_name == "write":
                            for kw in call.keywords:
                                if kw.arg == "content" and hasattr(kw.value, "value"):
                                    candidate = kw.value.value.strip()
                                    break

                        # Handle edit(...)
                        elif fn_name == "edit":
                            # Check edits=[{'oldText': ..., 'newText': ...}, ...]
                            working_code = current_code
                            has_edits = False

                            for kw in call.keywords:
                                if kw.arg == "edits" and isinstance(kw.value, ast.List):
                                    for elt in kw.value.elts:
                                        if isinstance(elt, ast.Dict):
                                            keys = [k.value for k in elt.keys if hasattr(k, "value")]
                                            vals = [v.value for v in elt.values if hasattr(v, "value")]
                                            d = dict(zip(keys, vals))
                                            old = d.get("oldText") or d.get("old_text") or ""
                                            new = d.get("newText") or d.get("new_text") or ""
                                            if old:
                                                working_code = _apply_search_replace_patch(working_code, old, new)
                                                has_edits = True

                                elif kw.arg in ("old_string", "old_text", "new_string", "new_text"):
                                    kwargs = {
                                        k.arg: k.value.value
                                        for k in call.keywords
                                        if hasattr(k.value, "value")
                                    }
                                    old = kwargs.get("old_string") or kwargs.get("old_text") or ""
                                    new = kwargs.get("new_string") or kwargs.get("new_text") or ""
                                    if old:
                                        working_code = _apply_search_replace_patch(working_code, old, new)
                                        has_edits = True

                                elif kw.arg == "updates" and hasattr(kw.value, "value"):
                                    updates_val = kw.value.value
                                    working_code = _parse_diff_updates(updates_val, working_code)
                                    has_edits = True

                            if has_edits:
                                candidate = working_code.strip()
                                break

                if candidate:
                    break
            except Exception:
                pass

            # Pattern C: Regex fallback for edits list if AST parse had minor syntax issue
            if not candidate and "edits=" in p_clean:
                working_code = current_code
                edit_dict_matches = re.findall(
                    r"\{\s*(?:['\"](?:oldText|old_text)['\"]\s*:\s*(['\"]{1,3})(.*?)\1)\s*,\s*(?:['\"](?:newText|new_text)['\"]\s*:\s*(['\"]{1,3})(.*?)\3)\s*\}",
                    p_clean,
                    re.DOTALL,
                )
                if edit_dict_matches:
                    for _, old_raw, _, new_raw in edit_dict_matches:
                        try:
                            old = old_raw.encode("utf-8").decode("unicode_escape")
                            new = new_raw.encode("utf-8").decode("unicode_escape")
                        except Exception:
                            old = old_raw
                            new = new_raw
                        working_code = _apply_search_replace_patch(working_code, old, new)
                    candidate = working_code.strip()
                    break

            # Pattern D: Regex fallback for updates=
            if not candidate and "updates=" in p_clean:
                upd_match = re.search(r"updates\s*=\s*(['\"]{1,3})(.*?)\1", p_clean, re.DOTALL)
                if upd_match:
                    updates_raw = upd_match.group(2)
                    try:
                        updates_text = updates_raw.encode("utf-8").decode("unicode_escape")
                    except Exception:
                        updates_text = updates_raw
                    candidate = _parse_diff_updates(updates_text, current_code).strip()
                    break

    # 4. If the text has tool call tokens (like read, read_file, edit_plan) with no code extracted,
    # NEVER treat raw tool-call tags as Python code!
    if not candidate:
        if "<|tool_call_start|>" in text or "<|tool_call_end|>" in text:
            logger.warning("Model emitted tool-call tags without valid code payload: %s", text[:80])
            return ""

        # Unclosed code fence recovery
        unclosed = re.search(r"```(?:python)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if unclosed:
            candidate = unclosed.group(1).strip()
        else:
            candidate = text.strip()

    # 5. AST validation: verify that candidate is valid, executable Python code
    if candidate:
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError as e:
            logger.warning("Extracted code failed AST parse validation: %s", e)
            return ""

    return ""


def truncate_traceback(stderr: str, max_lines: int = 10) -> str:
    """Filter and extract the most relevant lines of traceback or error logs.

    Extracts the last `max_lines` from error logs/test output containing
    the core Exception, preventing context window saturation on 2.6B models.
    """
    if not stderr or not stderr.strip():
        return ""

    lines = [line.rstrip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return ""

    if len(lines) <= max_lines:
        return "\n".join(lines)

    selected_lines = lines[-max_lines:]
    return "\n".join(selected_lines)
