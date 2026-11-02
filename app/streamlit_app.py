"""Chat with SBP regulations. Answers cite the exact regulation page they come from.

Run after building the index:
    streamlit run app/streamlit_app.py
"""

import re
import sys
from pathlib import Path

import anthropic
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.answer import ask  # noqa: E402
from assistant.retrieve import Retriever  # noqa: E402
from assistant.sources import DOCUMENTS, INDEX_DIR  # noqa: E402

st.set_page_config(page_title="SBP Regulations Assistant", layout="centered")

# Bullets in the SBP PDFs use private-use font glyphs that render as empty boxes.
PDF_SYMBOL = re.compile("[-]")


@st.cache_resource
def load_retriever() -> Retriever:
    return Retriever.load()


def show_sources(result: dict) -> None:
    for source in result.get("sources", []):
        with st.expander(f"[{source['n']}] {source['title']}, page {source['page']}"):
            for quote in source["quotes"]:
                st.markdown(f"> {quote}")
            st.markdown(f"[Open the regulation PDF]({source['url']}#page={source['page']})")


def show_passages(passages: list[dict], expanded: bool = False) -> None:
    if not passages:
        return
    with st.expander(f"Retrieved passages ({len(passages)})", expanded=expanded):
        for p in passages:
            st.markdown(f"**{p['title']}, page {p['page']}** (similarity {p['similarity']:.2f})")
            text = PDF_SYMBOL.sub("•", p["text"])
            st.caption(text[:600] + ("..." if len(text) > 600 else ""))


NO_KEY_NOTE = "No valid Anthropic API key found, so here are the most relevant passages instead of an answer."


def run_question(question: str, retriever: Retriever, k: int) -> dict:
    try:
        return ask(question, retriever, k=k)
    except TypeError as e:
        # The SDK raises TypeError (not an API error) when no credentials are configured at all.
        if "authentication method" not in str(e):
            raise
        note = NO_KEY_NOTE
    except anthropic.AuthenticationError:
        note = NO_KEY_NOTE
    except anthropic.RateLimitError:
        note = "The Claude API is rate limiting requests. Here are the most relevant passages; try again shortly."
    except anthropic.APIConnectionError:
        note = "Could not reach the Claude API. Here are the most relevant passages instead."
    except anthropic.AnthropicError as e:
        note = f"Claude API error ({type(e).__name__}). Here are the most relevant passages instead."
    return {"answer": note, "sources": [], "passages": retriever.search(question, k=k), "retrieval_only": True}


st.title("SBP Regulations Assistant")
st.caption(
    "Ask about State Bank of Pakistan rules on consumer financing, branchless banking, payment card "
    "security and AML/CFT. Answers come only from the regulation text and cite the page."
)

with st.sidebar:
    st.header("Documents")
    for doc in DOCUMENTS:
        st.markdown(f"- [{doc['title']}]({doc['url']})")
    k = st.slider("Passages sent to Claude", 3, 10, 5)
    st.caption("Without an ANTHROPIC_API_KEY the app still shows the most relevant passages.")
    st.caption("Study aid only, not legal or compliance advice.")

if not (INDEX_DIR / "chunks.jsonl").exists():
    st.error("No index found. Run `python scripts/download_docs.py` and `python -m assistant.ingest` first.")
    st.stop()

retriever = load_retriever()
st.session_state.setdefault("history", [])

for turn in st.session_state.history:
    st.chat_message("user").write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["result"]["answer"])
        show_sources(turn["result"])
        show_passages(turn["result"].get("passages", []))

question = st.chat_input("e.g. What is the maximum tenure for an auto loan?")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the regulations..."):
            result = run_question(question, retriever, k)
        st.write(result["answer"])
        show_sources(result)
        show_passages(result.get("passages", []), expanded=result.get("retrieval_only", False))
    st.session_state.history.append({"question": question, "result": result})
