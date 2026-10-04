"""
policy.py — the tool that reads the office's policy manual.

This is Section 8, and the point it has to land is this:

    RAG is not a different kind of magic. It is a TOOL.

Section 7 gave the agent `lookup_parcel` and it went and read a database.
Section 8 gives the same agent `search_policy` and it goes and reads a filing
cabinet. Same loop, same agent, one more entry in the tool list. That is the
whole idea, and building it as an ordinary tool alongside the others is what
makes it obvious rather than something we assert on a slide.

The retrieval machinery lives in `delorean/rag.py`. This file is only the
counter it is served over.

Why every result carries a citation
-----------------------------------
`search_policy` never returns bare text. Every hit comes back with the document
it came from and the section number within it, because an assessment office
cannot act on an answer it cannot trace. That is not politeness -- Section 10
depends on it, where the agent has to say *"Policy 4.2 bars the split"* and a
human has to be able to go and check.

It is also the direct answer to Section 4. There, the model invented an
ordinance and cited a Planning Commission that does not exist. Here it quotes a
real section of a real document, and you can open the file yourself.
"""

from __future__ import annotations

from .. import rag
from .base import fail, tool


@tool(
    name="search_policy",
    description=(
        "Search the Hill Valley County Assessor's Office policy manual and "
        "return the passages that answer a question. Use this for questions "
        "about office rules, procedures, deadlines, standards, or what is or "
        "is not permitted -- anything where the answer lives in a policy "
        "document rather than in a parcel record. Every result names the "
        "document and section it came from."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": ("The question to answer, in plain English, "
                                "e.g. 'what is the minimum lot size for a split?'"),
            },
            "top_k": {
                "type": "integer",
                "description": "How many passages to return. Default 3.",
            },
        },
        "required": ["question"],
    },
    name_aliases=("policy_search", "search_policies", "lookup_policy",
                  "search_manual", "policy_lookup", "search_documents",
                  "search_office_policy", "rag_search"),
    param_aliases={
        "question": ("query", "q", "text", "search", "question_text",
                     "search_query", "prompt"),
        "top_k": ("k", "n", "limit", "num_results", "top", "count", "n_results"),
    },
)
def search_policy(question: str, top_k: int = 3) -> dict | str:
    """Find the policy sections that answer a question."""
    question = (question or "").strip()
    if not question:
        return fail(
            "search_policy needs a question.",
            "Ask it the way a person would, for example "
            "'can a parcel be split if taxes are owed?'",
        )

    try:
        hits = rag.search(question, top_k=top_k)
    except FileNotFoundError as e:
        return fail(str(e), "The policy documents ship with the repository in "
                            "data/policy/. Check that folder exists.")

    if not hits:
        return fail(
            "Nothing in the policy manual matched that question.",
            "The manual covers parcel splits and combinations, sales "
            "qualification, ratio study standards, and the assessment "
            "calendar. Try rephrasing, or answer from general knowledge and "
            "say that the manual does not cover it.",
        )

    return {
        "question": question,
        "passages": [
            {
                "cite": f"{h['document']} §{h['section']}",
                "heading": h["heading"],
                "relevance": h["relevance"],
                "text": h["text"],
            }
            for h in hits
        ],
        # Said out loud so the model repeats it. An answer an appraiser cannot
        # trace back to a section number is not usable.
        "instruction": (
            "Answer using only these passages, and cite the section you used."
        ),
    }
