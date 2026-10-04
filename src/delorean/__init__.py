"""
delorean — the shared engine behind the 88mph mass-appraisal demos.

Everything in this package is imported by the notebooks in ../notebooks/.
The point of putting it here instead of copy-pasting into each notebook is
simple: the parcel lookup, ratio-study, and mapping tools get reused across
Sections 9 through 11. Six copies of the same function would drift apart
within a week.

Layout
------
    models.py      the model registry — which local model, and why
    llm.py         get_local_llm() / get_api_llm()
    tools/         the functions we hand to the agent
    rag/           document indexing and retrieval (Section 8)

Section 7 ("The Hoverboard") deliberately does NOT import from here.
There, the tool *is* the lesson, so the code stays inline in the notebook
where the audience can watch a plain Python function become something the
model is allowed to call.
"""

__version__ = "0.1.0"
