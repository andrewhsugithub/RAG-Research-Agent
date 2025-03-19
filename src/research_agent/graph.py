import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from research_agent.configuration import Configuration, SearchAPI
from research_agent.utils import (
    remove_think_tags,
    deduplicate_and_format_sources,
    format_sources,
    extract_rag_results,
    duckduckgo_search,
)
from research_agent.state import SummaryState, SummaryStateInput, SummaryStateOutput
from research_agent.prompts import (
    router_instructions,
    query_writer_instructions,
    web_summarizer_instructions,
    reflection_instructions,
)

from research_agent.rag import main as do_rag


# Nodes
def router(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0,
        format="json",
    )
    result = llm_json_mode.invoke(
        [
            SystemMessage(content=router_instructions),
            HumanMessage(content=state.research_topic),
        ]
    )
    query = json.loads(result.content)

    status = False
    if "datasource" not in query:
        status = "vectorstore" in str(query)
    else:
        status = query["datasource"] == "vectorstore"
    return {"rag_status": status}


def rag_decision(state: SummaryState) -> Literal["rag_research"]:
    # Return the node name you want to visit next
    if state.rag_status == True:
        return ["rag_research", "generate_query"]
    else:
        return ["generate_query"]


def generate_query(state: SummaryState, config: RunnableConfig):
    """Generate a query for web search"""

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(
        research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0,
        format="json",
    )
    result = llm_json_mode.invoke(
        [
            SystemMessage(content=query_writer_instructions_formatted),
            HumanMessage(content=f"Generate a query for web search:"),
        ]
    )
    query = json.loads(result.content)

    return {"search_query": query["query"]}


def rag_research(state: SummaryState, config: RunnableConfig):
    # TODO: move output_file, path, collection to config
    # output_file = "mock_rag_results.txt" # if want to use mock data
    output_file = "rag_results.txt"

    # if use mock data comment out the following line
    do_rag(
        query=state.research_topic,
        path="qdrant",
        collection="hf_docs",
        qdrant=True,
        chroma=False,
        output_file=output_file,
    )

    with open(output_file, "r") as f:
        rag_results = f.read()

    results, sources = extract_rag_results(rag_results)

    return {"rag_results": results, "rag_sources": [sources]}


def web_research(state: SummaryState, config: RunnableConfig):
    """Gather information from the web"""

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "duckduckgo")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.DUCKDUCKGO)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Search the web
    if search_api == "duckduckgo":
        search_results = duckduckgo_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=True
        )
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str],
    }


def summarize_sources(state: SummaryState, config: RunnableConfig):
    """Summarize the gathered sources"""

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Topic> \n {state.research_topic} \n <User Topic>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Topic> \n {state.research_topic} \n <User Topic>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0,
    )
    result = llm.invoke(
        [
            SystemMessage(content=web_summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    running_summary = result.content

    return {"running_summary": remove_think_tags(running_summary)}


def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """Reflect on the summary and generate a follow-up query"""

    human_message_content = (
        f"\n\n<Web Searcb Results> \n {state.running_summary} \n <Web Research Results>"
    )
    if state.rag_status == True:
        human_message_content += (
            f"\n\n<RAG Results> \n {state.rag_results} \n <RAG Results>"
        )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(
        base_url=configurable.ollama_base_url,
        model=configurable.local_llm,
        temperature=0,
        format="json",
    )
    result = llm_json_mode.invoke(
        [
            SystemMessage(
                content=reflection_instructions.format(
                    research_topic=state.research_topic
                )
            ),
            HumanMessage(
                content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {human_message_content}"
            ),
        ]
    )
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get("follow_up_query")

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query["follow_up_query"]}


def finalize_summary(state: SummaryState, config: RunnableConfig):
    """Finalize the summary"""

    # Format all accumulated sources into a single bulleted list
    combined_sources = state.sources_gathered + state.rag_sources
    all_sources = "\n".join(combined_sources)

    state.running_summary = (
        f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    )
    return {"running_summary": state.running_summary}


def route_research(
    state: SummaryState, config: RunnableConfig
) -> Literal["finalize_summary", "web_research"]:
    """Route the research based on the follow-up query"""

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count < int(configurable.max_web_research_loops):
        return "web_research"
    else:
        return "finalize_summary"


# Add nodes and edges
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)
builder.add_node("router", router)
builder.add_node("rag_research", rag_research)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router", rag_decision, ["rag_research", "generate_query"]
)
builder.add_edge("rag_research", END)
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_conditional_edges("reflect_on_summary", route_research)
# builder.add_edge(["rag_research", "summarize_sources"], "reflect_on_summary")
builder.add_edge("summarize_sources", "reflect_on_summary")
# builder.add_edge("rag_research", "reflect_on_summary")
builder.add_edge("finalize_summary", END)

graph = builder.compile(debug=False)


output = graph.get_graph().draw_mermaid_png()
with open("output.png", "wb") as f:
    f.write(output)

# query = "What's Model Context Protocol?"
# query = "What are the FAANG companies?"
query = "How to create a custom huggingface pipeline object?"
results = graph.invoke({"research_topic": query})

file = f"output_{query}.md"
with open(file, "w") as f:
    f.write(results["running_summary"])
