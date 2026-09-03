"""LangGraph graph: question generation, human-in-the-loop answer capture,
evaluation, targeted probing, and report compilation."""
from functools import lru_cache

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from domains import next_domain
from prompts import evaluation_prompt, probe_prompt, question_prompt
from state import InterviewState


class QuestionAndRubric(BaseModel):
    question: str = Field(description="One interview question, conceptual only")
    rubric: list[str] = Field(description="3 to 4 expected concepts")


class CoveredConcepts(BaseModel):
    covered: list[str] = Field(description="Rubric concepts the answer actually covered")


class ProbeQuestion(BaseModel):
    question: str = Field(description="One targeted follow-up question")


# Separate providers for question generation and evaluation so the model
# grading an answer isn't the same one that wrote the question and rubric.
@lru_cache
def _question_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7).with_structured_output(QuestionAndRubric)


@lru_cache
def _probe_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7).with_structured_output(ProbeQuestion)


@lru_cache
def _eval_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0).with_structured_output(
        CoveredConcepts, method="json_mode"
    )


def generate_question(state: InterviewState) -> dict:
    domain = next_domain(state["domains"], state["domain_pointer"])
    result = _question_llm().invoke(question_prompt(domain["name"], domain["description"]))
    return {
        "current_question": result.question,
        "current_rubric": result.rubric,
        "current_domain": domain["name"],
        "probe_count": 0,
        "covered_concepts": [],
        "current_answers": [],
        "domain_pointer": state["domain_pointer"] + 1,
    }


def human_answer(state: InterviewState) -> dict:
    answer = interrupt({
        "question": state["current_question"],
        "domain": state["current_domain"],
        "probe_count": state["probe_count"],
    })
    return {
        "human_answer": answer,
        "current_answers": state["current_answers"] + [answer],
    }


def evaluate_answer(state: InterviewState) -> dict:
    result = _eval_llm().invoke(
        evaluation_prompt(state["current_question"], state["current_rubric"], state["human_answer"])
    )
    covered = list(dict.fromkeys(state["covered_concepts"] + result.covered))
    return {"covered_concepts": covered}


def targeted_probe(state: InterviewState) -> dict:
    gap = [c for c in state["current_rubric"] if c not in state["covered_concepts"]]
    result = _probe_llm().invoke(probe_prompt(state["current_question"], gap, state["human_answer"]))
    return {
        "current_question": result.question,
        "probe_count": state["probe_count"] + 1,
    }


def route_after_evaluate(state: InterviewState) -> str:
    gap = [c for c in state["current_rubric"] if c not in state["covered_concepts"]]
    if gap and state["probe_count"] < state["max_probes"]:
        return "targeted_probe"
    return "finalize_question"


def finalize_question(state: InterviewState) -> dict:
    total = len(state["current_rubric"]) or 1
    hit = state["covered_concepts"]
    missed = [c for c in state["current_rubric"] if c not in hit]
    entry = {
        "question": state["current_question"],
        "domain": state["current_domain"],
        "answers": state["current_answers"],
        "concepts_hit": hit,
        "concepts_missed": missed,
        "score": len(hit) / total,
    }
    log = state["session_log"] + [entry]
    update = {"session_log": log}
    if state["question_index"] < state["num_questions"] - 1:
        update["question_index"] = state["question_index"] + 1
    return update


def route_after_finalize(state: InterviewState) -> str:
    return "generate_question" if len(state["session_log"]) < state["num_questions"] else "compile_report"


def compile_report(state: InterviewState) -> dict:
    log = state["session_log"]
    scores = [entry["score"] for entry in log]
    average = sum(scores) / len(scores) if scores else 0.0

    missed_counts: dict[str, int] = {}
    for entry in log:
        for concept in entry["concepts_missed"]:
            missed_counts[concept] = missed_counts.get(concept, 0) + 1
    weakest = sorted(missed_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "report_summary": {
            "average_score": average,
            "num_questions": len(log),
            "weakest_concepts": [concept for concept, _ in weakest],
        }
    }


def build_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("generate_question", generate_question)
    graph.add_node("human_answer", human_answer)
    graph.add_node("evaluate_answer", evaluate_answer)
    graph.add_node("targeted_probe", targeted_probe)
    graph.add_node("finalize_question", finalize_question)
    graph.add_node("compile_report", compile_report)

    graph.set_entry_point("generate_question")
    graph.add_edge("generate_question", "human_answer")
    graph.add_edge("human_answer", "evaluate_answer")
    graph.add_conditional_edges(
        "evaluate_answer", route_after_evaluate, ["targeted_probe", "finalize_question"]
    )
    graph.add_edge("targeted_probe", "human_answer")
    graph.add_conditional_edges(
        "finalize_question", route_after_finalize, ["generate_question", "compile_report"]
    )
    graph.add_edge("compile_report", END)

    return graph.compile(checkpointer=MemorySaver())
