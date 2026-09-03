"""Domain definitions for the interview session."""
import random

DOMAINS = [
    {
        "id": "python",
        "name": "Python",
        "description": "Mutability, the GIL, async/await, generators, decorators — conceptual only, no code.",
    },
    {
        "id": "langgraph",
        "name": "LangGraph",
        "description": "State, checkpointers, interrupt/human-in-the-loop, conditional edges.",
    },
    {
        "id": "langchain",
        "name": "LangChain",
        "description": "Chains vs agents, memory, retrievers, tool calling.",
    },
    {
        "id": "llm_fundamentals",
        "name": "LLM fundamentals",
        "description": "Transformers, attention, tokenization, embeddings, sampling.",
    },
    {
        "id": "rag",
        "name": "RAG and vector search",
        "description": "Chunking, embedding choice, indexing, hybrid search, retrieval evaluation.",
    },
    {
        "id": "prompt_eng",
        "name": "Prompt engineering and evaluation",
        "description": "Few-shot prompting, chain-of-thought, structured output, hallucination, eval sets.",
    },
    {
        "id": "system_design",
        "name": "LLM system design and agent architecture",
        "description": "Multi-agent patterns, tool design, latency and cost trade-offs.",
    },
    {
        "id": "production",
        "name": "Production AI engineering",
        "description": "Deployment, observability, caching, cost control, model fallback.",
    },
    {
        "id": "random",
        "name": "Random",
        "description": "Mixes questions across all domains above.",
    },
]

DOMAIN_LOOKUP = {d["id"]: d for d in DOMAINS}
NON_RANDOM_DOMAIN_IDS = [d["id"] for d in DOMAINS if d["id"] != "random"]


def next_domain(selected_domains: list[str], pointer: int) -> dict:
    """Round-robin through the user's selected domains; expand "random" per-draw
    so it doesn't get stuck repeating whatever domain happened to follow it."""
    domain_id = selected_domains[pointer % len(selected_domains)]
    if domain_id == "random":
        domain_id = random.choice(NON_RANDOM_DOMAIN_IDS)
    return DOMAIN_LOOKUP[domain_id]
