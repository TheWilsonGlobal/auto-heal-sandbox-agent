"""Streamlit Web UI for Autonomous Code-Healing Agent."""

import sys
import json
import urllib.request
import importlib
from datetime import datetime
from pathlib import Path
import streamlit as st
from langchain_openai import ChatOpenAI

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure live module reloads on Streamlit reruns
import src.config
import src.utils.parser
import src.agent.nodes
import src.agent.graph

importlib.reload(src.config)
importlib.reload(src.utils.parser)
importlib.reload(src.agent.nodes)
importlib.reload(src.agent.graph)

from src.config import settings
from src.agent.graph import build_healing_graph
from src.agent.state import AgentState
from run_demo import MockLFM25LLM
from constants import (
    BASIC_BUGGY_CODE,
    BASIC_TEST_CODE,
    BASIC_TASK_DESC,
    ADVANCED_BUGGY_CODE,
    ADVANCED_TEST_CODE,
    ADVANCED_TASK_DESC,
    CUSTOM_CSS,
)

st.set_page_config(
    page_title="Auto-Heal Sandbox Agent",
    page_icon="🛡️",
    layout="wide",
)

# Apply custom CSS styling for uniform columns and button heights
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🛡️ Autonomous Code-Healing Agent")
st.caption("Docker Sandbox + Local LiquidAI LFM 2.5 (2.6B) / Gemini + LangGraph Loop")


def check_endpoint_live(endpoint: str) -> bool:
    """Check if the OpenAI-compatible endpoint responds."""
    try:
        req = urllib.request.Request(f"{endpoint}/models", headers={"User-Agent": "python"})
        with urllib.request.urlopen(req, timeout=1.5):
            return True
    except Exception:
        return False


# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
provider_options = ["gemini", "openai"]
default_provider_idx = 0 if getattr(settings, "LLM_PROVIDER", "gemini").lower() == "gemini" else 1
llm_provider = st.sidebar.selectbox("LLM Provider", options=provider_options, index=default_provider_idx)

if llm_provider == "gemini":
    gemini_model = st.sidebar.text_input("Gemini Model", value=getattr(settings, "LLM_MODEL", "gemini-3.5-flash-lite"))
    gemini_key = st.sidebar.text_input("Gemini API Key", value=getattr(settings, "GEMINI_LLM_API_KEY", ""), type="password")
    max_retries = st.sidebar.slider("Max Retries", min_value=1, max_value=8, value=settings.MAX_RETRIES)
    use_mock_if_offline = st.sidebar.checkbox("Auto-fallback to Mock LLM if offline", value=False)

    if gemini_key:
        st.sidebar.success(f"Configured Gemini Provider:\n{gemini_model}")
    else:
        st.sidebar.warning("Please provide Gemini API Key")
else:
    llm_endpoint = st.sidebar.text_input("LLM Base URL", value=settings.LLM_BASE_URL)
    llm_model = st.sidebar.text_input("Model Name", value=settings.LLM_MODEL_NAME)
    max_retries = st.sidebar.slider("Max Retries", min_value=1, max_value=8, value=settings.MAX_RETRIES)
    use_mock_if_offline = st.sidebar.checkbox("Auto-fallback to Mock LLM if offline", value=True)

    # Status check for local LLM
    is_llm_live = check_endpoint_live(llm_endpoint)
    if is_llm_live:
        st.sidebar.success(f"Connected to local LLM:\n{llm_endpoint}")
    else:
        st.sidebar.warning(f"Local LLM offline at:\n{llm_endpoint}")

# Button row: Presets, Run Agent, Expand/Collapse
col_btn1, col_btn2, col_btn_run, col_btn3, col_btn4 = st.columns([2.2, 2.2, 2.2, 1.3, 1.3])
with col_btn1:
    if st.button("📦 Load Basic Bug (Calculator)", use_container_width=True):
        st.session_state["source_code_area"] = BASIC_BUGGY_CODE
        st.session_state["test_code_area"] = BASIC_TEST_CODE
        st.session_state["task_desc_area"] = BASIC_TASK_DESC
        st.session_state["inputs_expanded"] = True
        st.rerun()

with col_btn2:
    if st.button("🔥 Load Advanced Bug (LRU Cache)", use_container_width=True):
        st.session_state["source_code_area"] = ADVANCED_BUGGY_CODE
        st.session_state["test_code_area"] = ADVANCED_TEST_CODE
        st.session_state["task_desc_area"] = ADVANCED_TASK_DESC
        st.session_state["inputs_expanded"] = True
        st.rerun()

with col_btn_run:
    run_button = st.button("🚀 Run Healing Agent", type="primary", use_container_width=True)

with col_btn3:
    if st.button("🔽 Expand All", use_container_width=True):
        st.session_state["inputs_expanded"] = True
        st.session_state["result_expanded"] = True
        st.session_state["inspector_expanded"] = True
        st.session_state["export_expanded"] = True
        st.rerun()

with col_btn4:
    if st.button("🔼 Collapse All", use_container_width=True):
        st.session_state["inputs_expanded"] = False
        st.session_state["result_expanded"] = False
        st.session_state["inspector_expanded"] = False
        st.session_state["export_expanded"] = False
        st.rerun()

# 1. Buggy Code, Tests & Requirements (Unified Collapsible Section)
with st.expander("📁 1. Buggy Code, Tests & Requirements", expanded=st.session_state.get("inputs_expanded", True)):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Buggy Python Code")
        st.text_area(
            "Source Code (solution.py)",
            value=st.session_state.get("source_code_area", BASIC_BUGGY_CODE),
            height=480,
            key="source_code_area",
        )

    with col2:
        st.subheader("Pytest Test Cases")
        st.text_area(
            "Pytest Suite (test_suite.py)",
            value=st.session_state.get("test_code_area", BASIC_TEST_CODE),
            height=480,
            key="test_code_area",
        )

    with col3:
        st.subheader("Requirements")
        st.text_area(
            "Requirements & Guidelines",
            value=st.session_state.get("task_desc_area", BASIC_TASK_DESC),
            height=480,
            key="task_desc_area",
        )

# Always pull current input values from session state
source_code = st.session_state.get("source_code_area", BASIC_BUGGY_CODE)
test_code = st.session_state.get("test_code_area", BASIC_TEST_CODE)
task_desc = st.session_state.get("task_desc_area", BASIC_TASK_DESC)


if run_button:
    if not source_code.strip() or not test_code.strip():
        st.error("Please provide both source code and unit tests.")
        st.stop()

    custom_llm = None
    if llm_provider == "gemini":
        if not gemini_key.strip():
            st.error("Please enter a valid Gemini API Key in the sidebar.")
            st.stop()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            custom_llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key=gemini_key.strip(),
                temperature=settings.LLM_TEMPERATURE,
            )
        except Exception as e:
            st.error(f"Failed to initialize Gemini model: {e}")
            st.stop()
    else:
        if is_llm_live:
            custom_llm = ChatOpenAI(
                base_url=llm_endpoint,
                model=llm_model,
                api_key=settings.LLM_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
            )
        else:
            if use_mock_if_offline:
                st.info("Local LLM not detected. Using Mock LFM 2.5 LLM simulator to demonstrate the execution loop.")
                custom_llm = MockLFM25LLM()
            else:
                st.error(f"Cannot run: Local LLM at {llm_endpoint} is offline and mock fallback is disabled.")
                st.stop()

    agent = build_healing_graph(llm=custom_llm)

    state: AgentState = {
        "task_description": task_desc,
        "source_code": source_code,
        "test_code": test_code,
        "error_trace": None,
        "fixed_code": None,
        "iterations": 0,
        "max_retries": max_retries,
        "is_solved": False,
        "iteration_history": [],
    }

    st.write("---")
    log_history = []
    with st.status("🔄 Executing Autonomous Healing Agent...", expanded=True) as status_box:
        log_container = st.empty()


        def add_log(msg: str):
            ts = datetime.now().strftime("%H:%M:%S")
            entry = f"[{ts}] {msg}"
            log_history.append(entry)
            log_container.text("\n".join(log_history))

        add_log("[START] Initializing LangGraph stateful cyclic graph...")

        # Stream LangGraph events
        for event in agent.stream(state):
            for node_name, node_output in event.items():
                if node_name == "analyze_and_patch":
                    patch_len = len(node_output.get("fixed_code", ""))
                    add_log(f"🧠 [Planner/Writer] Node 'analyze_and_patch' finished. Generated patch ({patch_len} bytes).")
                elif node_name == "execute_in_sandbox":
                    exit_code = node_output.get("_exec_exit_code")
                    stdout_preview = (node_output.get("_exec_stdout", "").splitlines() or [""])[-1]
                    add_log(f"📦 [Docker Sandbox] Node 'execute_in_sandbox' finished. Exit code: {exit_code} ({stdout_preview})")
                elif node_name == "evaluate_result":
                    is_solved = node_output.get("is_solved")
                    iters = node_output.get("iterations")
                    if is_solved:
                        add_log(f"✅ [Evaluator] Node 'evaluate_result': Tests passed! Loop successfully terminated.")
                    else:
                        add_log(f"🔁 [Evaluator] Node 'evaluate_result': Tests failed (Iter {iters}/{max_retries}). Routing to patch node.")

                # Update current state
                state.update(node_output)

        if state.get("is_solved"):
            status_box.update(label="Healing Completed Successfully! 🎉", state="complete", expanded=False)
        else:
            status_box.update(label=f"Healing Exceeded Max Retries ({max_retries}) ⚠️", state="error", expanded=False)

    # Save run data to session_state
    run_record = {
        "timestamp": datetime.now().isoformat(),
        "task_description": task_desc,
        "source_code": source_code,
        "test_code": test_code,
        "is_solved": state.get("is_solved", False),
        "iterations": state.get("iterations", 0),
        "max_retries": max_retries,
        "final_traceback": state.get("error_trace", ""),
        "fixed_code": state.get("fixed_code", ""),
        "execution_logs": list(log_history),
        "iteration_history": list(state.get("iteration_history") or []),
    }
    st.session_state["latest_run"] = run_record
    st.session_state["result_expanded"] = True

# Display Results if a run has occurred
if "latest_run" in st.session_state:
    run_record = st.session_state["latest_run"]

    # 2. Healed Result & Patched Code (Collapsible)
    with st.expander("📁 2. Healed Result & Patched Code", expanded=st.session_state.get("result_expanded", True)):
        if run_record.get("is_solved"):
            st.success("✨ Bug healed successfully! Code passed all unit tests in the Docker sandbox.")
            st.code(run_record.get("fixed_code", ""), language="python")
        else:
            st.error("Failed to heal code within the allocated retry budget.")
            if run_record.get("final_traceback"):
                st.subheader("Final Traceback:")
                st.code(run_record.get("final_traceback", ""), language="text")
            if run_record.get("fixed_code"):
                st.subheader("Last Generated Patch:")
                st.code(run_record.get("fixed_code", ""), language="python")

    # 3. Detailed Iteration Inspector (Collapsible)
    iterations = run_record.get("iteration_history", [])
    if iterations:
        with st.expander(
            f"🔍 3. Detailed Iteration Inspector ({len(iterations)} attempts recorded)",
            expanded=st.session_state.get("inspector_expanded", False),
        ):
            for item in iterations:
                iter_num = item.get("iteration")
                status_icon = "✅ Passed" if item.get("passed") else "❌ Failed"
                st.markdown(f"#### Iteration {iter_num} — {status_icon} (Exit code: {item.get('exit_code')})")

                col_it1, col_it2 = st.columns(2)
                with col_it1:
                    st.caption("Pytest Sandbox Output:")
                    st.code(item.get("stdout") or item.get("stderr") or "No output", language="text")
                with col_it2:
                    st.caption("Model Response / Tool Output:")
                    st.code(item.get("raw_llm_response") or "N/A", language="text")

                st.caption("Extracted Code Patch:")
                st.code(item.get("fixed_code") or "N/A", language="python")
                st.write("---")


    # Format download payloads
    json_payload = json.dumps(run_record, indent=2, ensure_ascii=False)

    log_lines = [
        "=" * 70,
        "             AUTONOMOUS CODE-HEALING AGENT EXECUTION LOG",
        "=" * 70,
        f"Timestamp:        {run_record.get('timestamp')}",
        f"Task Description: {run_record.get('task_description')}",
        f"Solved Status:    {run_record.get('is_solved')}",
        f"Iterations:       {run_record.get('iterations')} / {run_record.get('max_retries')}",
        "",
        "-" * 50,
        "1. BUGGY PYTHON CODE",
        "-" * 50,
        run_record.get("source_code", ""),
        "",
        "-" * 50,
        "2. PYTEST TEST CASES",
        "-" * 50,
        run_record.get("test_code", ""),
        "",
        "-" * 50,
        "3. EXECUTION CHECKPOINT LOGS",
        "-" * 50,
        "\n".join(run_record.get("execution_logs", [])),
        "",
        "-" * 50,
        "4. FINAL TRACEBACK",
        "-" * 50,
        run_record.get("final_traceback") or "None (All tests passed)",
        "",
        "-" * 50,
        "5. FINAL PATCHED / HEALED CODE",
        "-" * 50,
        run_record.get("fixed_code") or "None",
        "",
        "=" * 70,
        "DETAILED ITERATION BREAKDOWN",
        "=" * 70,
    ]

    for item in run_record.get("iteration_history", []):
        it_num = item.get("iteration")
        it_status = "PASSED" if item.get("passed") else "FAILED"
        log_lines.extend([
            f"\n>>> ITERATION {it_num} [{it_status}] (Exit Code: {item.get('exit_code')}) <<<",
            "--- Raw LLM Response ---",
            item.get("raw_llm_response") or "N/A",
            "--- Pytest Stdout ---",
            item.get("stdout") or "None",
            "--- Pytest Stderr ---",
            item.get("stderr") or "None",
            "--- Extracted Code ---",
            item.get("fixed_code") or "None",
            "-" * 40,
        ])

    log_lines.append("=" * 70)
    txt_log_payload = "\n".join(log_lines)

    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 4. Export Logs & Artifacts (Collapsible)
    with st.expander("📥 4. Export Logs & Artifacts", expanded=st.session_state.get("export_expanded", True)):
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="💾 Download Session Report (.json)",
                data=json_payload,
                file_name=f"healing_report_{ts_slug}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                label="📄 Download Execution Log (.log)",
                data=txt_log_payload,
                file_name=f"healing_log_{ts_slug}.log",
                mime="text/plain",
                use_container_width=True,
            )

