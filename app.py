"""Local Streamlit interface. Run: python -m streamlit run app.py"""

import io
import json
from pathlib import Path

import streamlit as st
from agent import Agent, Analysis, Memory

st.set_page_config(
    page_title="Lens | Data analysis agent", page_icon="🔎", layout="wide"
)
st.title("Lens")
st.caption("Supermarket sales analysis with Llama 3 and pandas.")
with st.sidebar:
    st.header("Dataset & runtime")
    uploaded = st.file_uploader("Upload UTF-8 CSV (10 MB maximum)", type=["csv"])
    st.caption("Default dataset: SuperMarket Analysis.csv")
    offline = st.toggle("Offline mode", value=False)
    st.caption(
        "Offline uses a limited rule parser. LLM mode uses Ollama with llama3:8b. Start Ollama before asking a question."
    )
    scope_override = st.text_input(
        "Optional shared memory scope",
        help="Leave blank for content fingerprint isolation. Reuse a scope only for compatible versions of the same dataset.",
    )
    st.caption(
        "Memory is saved locally in memory.sqlite. Raw datasets and chat logs are not saved."
    )

raw = (
    uploaded.getvalue()
    if uploaded
    else (Path(__file__).parent / "data" / "SuperMarket Analysis.csv").read_bytes()
)
try:
    analysis = Analysis(io.BytesIO(raw), Memory(), scope_override or None)
except Exception as e:
    st.error(str(e))
    st.stop()
identity = analysis.scope
if st.session_state.get("dataset") != identity:
    st.session_state["dataset"] = identity
    st.session_state["answers"] = []
a, b, c = st.columns(3)
a.metric("Rows", len(analysis.df))
b.metric("Columns", len(analysis.df.columns))
c.metric("Mode", "Offline parser" if offline else "LLM + local tools")
st.caption(f"Memory scope: {identity}")
with st.expander("Preview and data quality"):
    st.dataframe(analysis.df.head(20), use_container_width=True)
    st.json(analysis.profile())

tab_ask, tab_memory = st.tabs(["Analysis", "Persistent memory"])
with tab_memory:
    st.write(
        "Save only reusable definitions, preferences or explicit named filters. Saving the same name replaces its old value."
    )
    with st.form("memory"):
        name = st.text_input("Name", placeholder="members")
        kind = st.selectbox("Kind", ["filter", "definition", "preference"])
        text = st.text_input(
            "Meaning or preference", placeholder="Member customers; or monthly"
        )
        clauses = st.text_area(
            "Filter clauses (JSON, for filter kind only)",
            '[{"column":"Customer type","op":"eq","value":"Member"}]',
        )
        if st.form_submit_button("Save memory"):
            try:
                result = analysis.call(
                    "remember",
                    {
                        "name": name,
                        "kind": kind,
                        "text": text,
                        "filters": json.loads(clauses) if kind == "filter" else [],
                    },
                )
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(
                        "Saved. Future questions in this scope retrieve this fact."
                    )
            except ValueError as e:
                st.error(str(e))
    facts = analysis.memory.list(identity)
    st.json(facts)
    if facts:
        selected = st.selectbox("Fact to remove", [f["name"] for f in facts])
        if st.button("Forget selected fact"):
            analysis.memory.delete(identity, selected)
            st.rerun()

with tab_ask:
    st.caption(
        "Try: Total Sales by City • Average Rating by Product line • Monthly Sales trend by Date"
    )
    question = st.chat_input("Ask a question using your column names")
    if question:
        with st.spinner("Running analysis tools…"):
            try:
                response = Agent(analysis, offline=offline).ask(question)
                st.session_state["answers"].append((question, response))
            except Exception as e:
                st.error(f"Could not complete the request: {e}")
    for question, response in st.session_state["answers"]:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(response["answer"])
            with st.expander("Tool evidence: inputs, outputs, trace IDs"):
                st.json(response["trace"])
                for t in response["trace"]:
                    r = t["result"]
                    if t["tool"] == "trend" and r.get("buckets"):
                        import pandas as pd

                        st.line_chart(
                            pd.DataFrame(r["buckets"]).set_index("date")["value"]
                        )
                    if (
                        t["tool"] == "aggregate"
                        and r.get("group_by")
                        and r.get("values")
                    ):
                        import pandas as pd

                        st.bar_chart(
                            pd.DataFrame(r["values"]).set_index(r["group_by"])["value"]
                        )
            st.download_button(
                "Download evidence JSON",
                json.dumps(response, indent=2),
                file_name="analysis.json",
                key=f'download_{len(question)}_{st.session_state["answers"].index((question,response))}',
            )
