# Custom RAG Agent with LangGraph

An agentic RAG system built with [LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag) that decides *when* to retrieve, grades what it retrieved, and self-corrects — instead of blindly retrieving on every question.

![LangGraph Hybrid RAG Tutorial-Data Flow](https://mintcdn.com/langchain-5e9cc07a/I6RpA28iE233vhYX/images/langgraph-hybrid-rag-tutorial.png?w=1650&fit=max&auto=format&n=I6RpA28iE233vhYX&q=85&s=4e5f030034237589f651b704d0377a76)

Based on the official LangGraph agentic RAG tutorial, with these substitutions and additions:

- **LLM**: Nemotron (`nemotron-3-ultra-free`) via the [Zen API](https://opencode.ai/zen) (OpenAI-compatible) instead of OpenAI
- **Embeddings**: [Jina AI](https://jina.ai/embeddings/) (`jina-embeddings-v3`) instead of OpenAI
- **Vector store**: persistent [Chroma](https://www.trychroma.com/) instead of in-memory — indexes once, reuses across runs
- **Loop cap**: question rewriting is limited to 2 retries, so bad retrievals answer with best effort instead of looping forever
- **CLI**: one-shot questions, interactive REPL, and `--reindex`

## How it works

The graph decides at each step:

1. `generate_query_or_respond` — answer directly, or call the retriever tool
2. `retrieve` — semantic search over the indexed documents (Lilian Weng's blog posts on reward hacking, hallucination, and diffusion video models)
3. `grade_documents` — an LLM grades retrieved context for relevance (structured output with a plain-text fallback)
4. Relevant → `generate_answer`; irrelevant → `rewrite_question` and retry (max 2 rewrites)

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in your keys
```

| Variable | Purpose |
|----------|---------|
| `JINA_API_KEY` | Jina AI embeddings |
| `ZEN_API_KEY` | Zen API (LLM calls) |
| `ZEN_API_BASE` | Zen endpoint (default: `https://opencode.ai/zen/v1`) |

## Usage

```bash
# One-shot question (first run fetches, chunks, and indexes the documents)
uv run rag-agent "What does Lilian Weng say about types of reward hacking?"

# Interactive REPL
uv run rag-agent

# Force a rebuild of the vector store (after changing source URLs or chunk settings)
uv run rag-agent --reindex "your question"
```

The vector store persists in `./chroma_db/` (gitignored). Delete it or use `--reindex` to rebuild. There is no automatic staleness detection — if the source documents change, reindex manually.

## Development

```bash
uv run pytest        # unit tests (all LLM/embedding calls mocked, no network)
uv run ruff check    # lint
```

Project layout:

```
src/lg_custom_rag_agent/
  config.py           # env loading, API key accessors, model name
  fetch_documents.py  # fetch, chunk, embed, persist; retriever tool
  agent.py            # graph state, nodes, edges, compiled graph
  main.py             # CLI entry point (rag-agent)
tests/
  test_agent_nodes.py # node-level tests with mocked models
```

## Customization

- **Different documents**: edit `DEFAULT_URLS` in `fetch_documents.py`, then `--reindex`
- **Chunking**: `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` in `fetch_documents.py`, then `--reindex`
- **Model**: `ZEN_MODEL` in `config.py` (any model on your Zen endpoint)
- **Rewrite cap**: the `>= 2` check in `grade_documents` in `agent.py`
