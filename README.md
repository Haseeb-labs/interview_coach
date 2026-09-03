# Interview Coach

A mock interview app for AI engineering roles. It quizzes on conceptual and
theoretical topics only — no coding questions, no code execution — across
nine domains (Python internals, LangGraph, LangChain, LLM fundamentals, RAG,
prompt engineering, system design, production AI engineering, or a random
mix), and gives live feedback with targeted follow-up probing.

## Setup

```
git clone https://github.com/Haseeb-labs/interview_coach.git
cd interview_coach
pip install -r requirements.txt
copy .env.example .env   # then fill in OPENAI_API_KEY and GROQ_API_KEY
streamlit run app.py
```

## Architecture

The interview flow is a LangGraph graph, not a script: question generation
uses OpenAI, and answer evaluation uses Groq, kept as separate models so
the same model that wrote the question and rubric isn't the one grading the
answer against it. After each answer, the graph checks for rubric concepts
the candidate missed and can loop into a targeted follow-up question, capped
at a configurable number of probes per question so a weak answer can't stall
the session indefinitely. Waiting for the candidate's answer is a LangGraph
`interrupt()` — the graph pauses mid-run and the Streamlit UI resumes it with
the submitted answer on the next rerun, using a checkpointer keyed by a
per-session thread id to survive Streamlit's rerun-the-whole-script model.
