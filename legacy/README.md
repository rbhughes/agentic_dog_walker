# legacy/ — the 2024 LangChain implementation

Frozen reference, not live code. The ReAct-era agent (`dog_walker_2024/agent.py`),
its single-string tools, the Streamlit UI (`app.py`), and their tests moved here
when the 2026 rebuild began. Tool logic gets ported OUT of here into
`src/dog_walker/toolbox.py` piece by piece; nothing here is imported by new code.
