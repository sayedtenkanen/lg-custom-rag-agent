"""CLI runner for the LG Custom RAG Agent."""

import argparse
import sys

from langchain_core.messages import HumanMessage

from lg_custom_rag_agent.config import get_jina_api_key, get_zen_api_key


def main() -> None:
    """Entry point for the rag-agent CLI."""
    parser = argparse.ArgumentParser(
        prog="rag-agent",
        description="Agentic RAG powered by LangGraph",
    )
    parser.add_argument("question", nargs="?", default=None, help="Question to ask the agent")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-index of documents before answering",
    )
    args = parser.parse_args()

    # Validate API keys early with a friendly message
    try:
        get_jina_api_key()
    except ValueError:
        print("Error: JINA_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)
    try:
        get_zen_api_key()
    except ValueError:
        print("Error: ZEN_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    # Lazy imports — only fail after key check
    from lg_custom_rag_agent.fetch_documents import get_retriever

    if args.reindex:
        print("Re-indexing documents...")
        get_retriever(force_reindex=True)
        print("Done.")

    from lg_custom_rag_agent.agent import graph

    if args.question:
        _run_single(graph, args.question)
    else:
        _repl(graph)


def _run_single(graph, question: str) -> None:
    """Run the graph once and stream updates."""
    print(f"Question: {question}\n")
    answer = ""
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            msgs = update.get("messages", [])
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", "")
                if content:
                    print(f"[{node_name}] {content}")
                    answer = content
    print(f"\nAnswer: {answer}")


def _repl(graph) -> None:
    """Interactive REPL."""
    print("RAG Agent (type 'quit' or 'exit' to stop)\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question in ("quit", "exit"):
            break
        if not question:
            continue
        _run_single(graph, question)
        print()


if __name__ == "__main__":
    main()
