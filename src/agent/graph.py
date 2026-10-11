"""LangGraph assembly for the autonomous healing loop."""

from typing import Any, Literal, Optional
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import HealingNodes
from src.sandbox.docker_runner import DockerSandboxRunner


def route_next(state: AgentState) -> Literal["analyze_and_patch", "__end__"]:
    """Conditional router determining whether to terminate or continue iteration."""
    if state.get("is_solved", False):
        return END

    iterations = state.get("iterations", 0)
    max_retries = state.get("max_retries", 4)

    if iterations >= max_retries:
        return END

    return "analyze_and_patch"


def build_healing_graph(
    llm: Optional[Any] = None,
    sandbox: Optional[DockerSandboxRunner] = None,
) -> Any:
    """Build and compile the LangGraph cyclic workflow."""
    nodes = HealingNodes(llm=llm, sandbox=sandbox)

    workflow = StateGraph(AgentState)

    # Register micro-nodes
    workflow.add_node("analyze_and_patch", nodes.analyze_and_patch)
    workflow.add_node("execute_in_sandbox", nodes.execute_in_sandbox)
    workflow.add_node("evaluate_result", nodes.evaluate_result)

    # Define edges: analyze_and_patch -> execute_in_sandbox -> evaluate_result -> conditional
    workflow.set_entry_point("analyze_and_patch")
    workflow.add_edge("analyze_and_patch", "execute_in_sandbox")
    workflow.add_edge("execute_in_sandbox", "evaluate_result")
    workflow.add_conditional_edges(
        "evaluate_result",
        route_next,
        {
            "analyze_and_patch": "analyze_and_patch",
            END: END,
        },
    )

    return workflow.compile()


def create_healing_agent(
    llm: Optional[Any] = None,
    sandbox: Optional[DockerSandboxRunner] = None,
) -> Any:
    """Convenience factory to create an executable healing agent."""
    return build_healing_graph(llm=llm, sandbox=sandbox)
