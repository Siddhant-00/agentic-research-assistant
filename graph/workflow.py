from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState

from graph.nodes import (
    rewrite_query,
    route_question,
    retrieve_documents,
    generate_answer
)
from graph.nodes import (
    rewrite_query,
    route_question,
    retrieve_documents,
    rerank_retrieved_documents,
    generate_answer,
    verify_answer
)

memory = MemorySaver()

workflow = StateGraph(GraphState)

# Nodes
workflow.add_node(
    "rewrite_query",
    rewrite_query
)

workflow.add_node(
    "route_question",
    route_question
)

workflow.add_node(
    "retrieve_documents",
    retrieve_documents
)

workflow.add_node(
    "generate_answer",
    generate_answer
)

workflow.add_node(
    "rerank_documents",
    rerank_retrieved_documents
)

workflow.add_node(
    "verify_answer",
    verify_answer
)
# Edges
workflow.add_edge(
    START,
    "rewrite_query"
)

workflow.add_edge(
    "rewrite_query",
    "route_question"
)

workflow.add_edge(
    "route_question",
    "retrieve_documents"
)

workflow.add_edge(
    "retrieve_documents",
    "rerank_documents"
)

workflow.add_edge(
    "rerank_documents",
    "generate_answer"
)

workflow.add_edge(
    "generate_answer",
    "verify_answer"
)

workflow.add_edge(
    "verify_answer",
    END
)

app_graph = workflow.compile(
    checkpointer=memory
)