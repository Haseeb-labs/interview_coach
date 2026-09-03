"""Shared state schema passed between LangGraph nodes."""
from typing import TypedDict


class InterviewState(TypedDict):
    domains: list[str]
    num_questions: int
    question_index: int
    current_question: str
    current_rubric: list[str]
    covered_concepts: list[str]
    human_answer: str
    probe_count: int
    max_probes: int
    session_log: list[dict]

    # bookkeeping not in the original spec but needed to drive the graph
    current_domain: str
    current_answers: list[str]
    domain_pointer: int
    report_summary: dict


def initial_state(domains: list[str], num_questions: int, max_probes: int) -> InterviewState:
    return {
        "domains": domains,
        "num_questions": num_questions,
        "question_index": 0,
        "current_question": "",
        "current_rubric": [],
        "covered_concepts": [],
        "human_answer": "",
        "probe_count": 0,
        "max_probes": max_probes,
        "session_log": [],
        "current_domain": "",
        "current_answers": [],
        "domain_pointer": 0,
        "report_summary": {},
    }
