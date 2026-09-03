"""All LLM prompt templates, kept in one place."""

CONCEPTUAL_ONLY_RULE = (
    "Stay strictly conceptual and theoretical. Never ask the candidate to write, "
    "read, debug, or trace code. Never include a code snippet in your output."
)


def question_prompt(domain_name: str, domain_description: str) -> str:
    return (
        f"You are interviewing a candidate for an AI engineering role on the topic "
        f"of {domain_name}. Topic scope: {domain_description}\n\n"
        f"{CONCEPTUAL_ONLY_RULE}\n\n"
        "Write one interview question that probes conceptual understanding, "
        "and a rubric of 3 to 4 specific concepts a strong answer would cover. "
        "The question should be answerable in a few sentences of spoken explanation, "
        "not by writing code."
    )


def evaluation_prompt(question: str, rubric: list[str], answer: str) -> str:
    rubric_list = "\n".join(f"- {item}" for item in rubric)
    return (
        "You are grading a candidate's spoken answer in an AI engineering interview. "
        f"{CONCEPTUAL_ONLY_RULE}\n\n"
        f"Question:\n{question}\n\n"
        f"Rubric concepts expected in a strong answer:\n{rubric_list}\n\n"
        f"Candidate's answer:\n{answer}\n\n"
        "Respond with a JSON object with a single key \"covered\": a list of exactly "
        "which rubric concepts (copy the wording from the rubric) the answer actually "
        "covered. Do not credit a concept unless the answer genuinely demonstrates "
        "understanding of it. If the answer covers none of them, return an empty list."
    )


def probe_prompt(question: str, missing_concepts: list[str], answer: str) -> str:
    missing_list = "\n".join(f"- {item}" for item in missing_concepts)
    return (
        "You are conducting a follow-up in an AI engineering interview. "
        f"{CONCEPTUAL_ONLY_RULE}\n\n"
        f"Original question:\n{question}\n\n"
        f"Candidate's answer so far:\n{answer}\n\n"
        f"The answer missed these expected concepts:\n{missing_list}\n\n"
        "Ask one targeted follow-up question that pushes the candidate specifically "
        "toward the missing concept(s) above. Do not ask a generic 'tell me more' "
        "question — reference what is actually missing."
    )
