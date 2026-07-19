#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# sitsrag is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version; see the LICENSE file for more details.
#

"""LangGraph ReAct agent graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver

#
# Constants
#
SYSTEM_PROMPT = """
You are an expert assistant for the SITS package (R), which provides
an API for satellite image time series analysis - from cube creation to classification.

You have access to tools that search documentation, function references, research
articles, and ontology data (spectral indices, satellites, collections). Use them
to ground your answers.

TOOL USAGE GUIDELINES

- Use `search_documentation` for workflows, concepts, and usage patterns.
- Use `search_reference` for function signatures, parameters, and return types.
- Use `get_function_detail` to retrieve the COMPLETE documentation for a specific
  function when search_reference returned only partial results (e.g., you found the
  function name but need the full signature, all parameters, or examples).
- Use `search_articles` for real-world applications and case studies.
- Use `lookup_spectral_index` when a spectral index is mentioned (NDVI, EVI, etc.).
- Use `lookup_satellite` when a satellite is mentioned (Sentinel-2, Landsat-8, etc.).
- Use `lookup_collection` ALWAYS before writing code that uses band names.

STRICT RULES

FUNCTIONS AND PARAMETERS
- Only use functions and parameters that appear in tool results.
- NEVER invent a function, parameter, or argument.
- Use exact function names as they appear (e.g., `sits_train()`, `sits_classify()`).
- If search_reference returns partial info, call `get_function_detail` for the full docs.
- NEVER create an example using combinations you are not sure. If required, confirm the function signatures

BAND NAMES - HIGHEST PRIORITY
- Band names in code MUST come from `lookup_collection` results.
- Never use generic band names (NIR, Red, SWIR) in code.
- If the user does not specify a collection, ask which one before writing code.

  WRONG - uses generic name:
    sits_apply(data, NDVI = (NIR - Red) / (NIR + Red))

  CORRECT - uses exact band names from collection lookup:
    # Band names from SITS collection: Sentinel-2-L2A
    sits_apply(data, NDVI = (B08 - B04) / (B08 + B04))

SPECTRAL INDICES
- Use `lookup_spectral_index` for the formula and required band types.
- Then use `lookup_collection` to resolve actual band names for code.

PARTIAL ANSWERS
- If you can explain a concept but cannot write code (functions not in context),
  explain what you know and state what is missing.

UNKNOWNS
- If something is absent from tool results, say so directly.
- Never fill gaps with training knowledge about sits specifically.
- You MAY use general R/Python knowledge for syntax and standard library.
- Do NOT create facts about SITS you cannot confirm. For example, SITS apply
cloud removal automatically, so it is not required to invent code about this fact

RESPONSE FORMAT
- Answer in markdown with code blocks wrapped in ```r or ```python.
- Default to R unless the user specifies Python.
- Show complete, runnable code blocks.
- Add a comment indicating the collection source when using band names.
- We MUST keep explanations concise. Lead with code for technical questions.
- We MUST answer focusing on what the user ask. We can provide extra details,
but WE MUST BE MODERATED. IT IS NOT REQUIRED TO SHOW MULTIPLE EXAMPLES of the same thing

SOURCE CITATIONS
- When citing information from tool results, use the source markers (e.g., [S1], [S2]) provided in the tool output.
- Place markers inline next to the statement they support.

Current date: {current_date}
"""


#
# States
#
class AgentState(MessagesState, total=False):
    """Extended agent state with source citations."""

    sources: dict[str, Any] = {}
    """Source references keyed by marker (S1, S2...) from tool artifacts."""


#
# Nodes
#
async def sources_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Collect source citations from tool message artifacts.

    Args:
        state (AgentState): Agent state.
        config (RunnableConfig): Runnable configuration.

    Returns:
        dict[str, Any]: Source references keyed by marker (S1, S2...) from tool artifacts.
    """
    # Get tool messages
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]

    # Build sources
    sources = {}

    # Iterate over tool messages
    for msg in tool_messages:
        # Get artifact
        artifact = getattr(msg, "artifact", None)

        # Update sources
        if artifact and "sources" in artifact:
            sources.update(artifact["sources"])

    # Return sources
    return {"sources": sources}


#
# High-level interface
#
def build_agent_graph(
    llm: BaseChatModel,
    tools: list[Any],
    checkpointer: BaseCheckpointSaver,
    system_prompt: str,
):
    """Build a ReAct agent with tools and persistent memory.

    Args:
        llm: LangChain chat model (ChatLiteLLM).

        tools: List of agent tools (search + ontology lookups).

        checkpointer: LangGraph checkpointer (in-memory) for conversation state.

        system_prompt: System message that guides agent behavior.

    Returns:
        Compiled LangGraph agent graph.
    """
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SystemMessage(content=system_prompt),
    )

    async def agent_node(
        state,
        config,
    ):
        return await agent_graph.ainvoke(input=state, config=config)

    # Build graph
    builder = StateGraph(AgentState)
    _ = builder.add_node("agent", agent_node)
    _ = builder.add_node("sources", sources_node)
    _ = builder.add_edge(START, "agent")
    _ = builder.add_edge("agent", "sources")
    _ = builder.add_edge("sources", END)

    # Compile!
    return builder.compile(checkpointer=checkpointer)
