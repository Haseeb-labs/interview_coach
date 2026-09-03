"""Streamlit entrypoint for the mock interview app."""
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from domains import DOMAINS
from graph import build_graph
from state import initial_state

load_dotenv()

st.set_page_config(page_title="AI Engineering Mock Interview", layout="centered")


@st.cache_resource
def get_graph():
    return build_graph()


def run_until_pause(resume_value=None):
    """Advance the graph and sync session_state with wherever it stops next
    (an interrupt waiting on an answer, or the compiled report)."""
    graph = get_graph()
    config = st.session_state.thread_config
    if resume_value is not None:
        graph.invoke(Command(resume=resume_value), config=config)
    else:
        graph.invoke(st.session_state.init_input, config=config)

    snapshot = graph.get_state(config)
    st.session_state.graph_values = snapshot.values
    if snapshot.next:
        st.session_state.stage = "question"
    else:
        st.session_state.stage = "report"


def start_session(domains, num_questions, max_probes):
    thread_id = str(uuid.uuid4())
    st.session_state.thread_config = {"configurable": {"thread_id": thread_id}}
    st.session_state.init_input = initial_state(domains, num_questions, max_probes)
    run_until_pause()


def render_start():
    st.title("AI Engineering Mock Interview")
    st.write("Pick the domains to be quizzed on. Questions are conceptual only.")

    options = [d["id"] for d in DOMAINS]
    labels = {d["id"]: d["name"] for d in DOMAINS}
    selected = st.multiselect(
        "Domains", options=options, default=["random"], format_func=lambda x: labels[x]
    )
    num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)
    max_probes = st.number_input("Max follow-up probes per question", min_value=0, max_value=5, value=2)

    if st.button("Start", disabled=not selected):
        start_session(selected, int(num_questions), int(max_probes))
        st.rerun()


def render_question():
    values = st.session_state.graph_values
    total = values["num_questions"]
    index = values["question_index"]
    st.progress((index) / total)
    st.caption(f"Question {index + 1} of {total} — {values['current_domain']}")
    if values["probe_count"] > 0:
        st.caption(f"Follow-up {values['probe_count']} of {values['max_probes']}")

    st.subheader(values["current_question"])
    answer = st.text_area("Your answer", key=f"answer_{index}_{values['probe_count']}")

    if st.button("Submit answer"):
        if answer.strip():
            run_until_pause(resume_value=answer)
            st.rerun()
        else:
            st.warning("Enter an answer before submitting.")


def render_report():
    values = st.session_state.graph_values
    summary = values["report_summary"]
    average = summary["average_score"] * 100
    weakest = ", ".join(summary["weakest_concepts"]) if summary["weakest_concepts"] else "none"
    st.title("Interview report")
    st.write(f"{average:.0f}% average across {summary['num_questions']} questions. Weakest area: {weakest}.")

    for i, entry in enumerate(values["session_log"], start=1):
        with st.expander(f"Q{i} ({entry['domain']}) — {entry['score'] * 100:.0f}%"):
            st.write(entry["question"])
            if entry["concepts_missed"]:
                st.write("Missed: " + ", ".join(entry["concepts_missed"]))
            else:
                st.write("All rubric concepts covered.")

    if st.button("Start new session"):
        for key in ("thread_config", "init_input", "graph_values", "stage"):
            st.session_state.pop(key, None)
        st.rerun()


if "stage" not in st.session_state:
    st.session_state.stage = "start"

if st.session_state.stage == "start":
    render_start()
elif st.session_state.stage == "question":
    render_question()
else:
    render_report()
