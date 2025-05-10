__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from agent import load_documents, create_vectorstore, route_question

st.set_page_config(page_title="RAG-Powered Multi-Agent Assistant", layout="centered")
st.title("RAG-Powered Multi-Agent Assistant")

st.markdown(
    """
    This assistant uses:
    - Calculator tool for math
    - Dictionary tool for word definitions
    - RAG tool for Q&A over documents
    """
)

if "vectordb" not in st.session_state:
    with st.spinner("Indexing documents..."):
        docs = load_documents()
        st.session_state.vectordb = create_vectorstore(docs)
        st.success("Documents indexed!")

question = st.text_input("Ask a question (e.g. 'What is your return policy?', 'Calculate 8 * 4', 'Define resilient')")

if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        try:
            answer = route_question(question, st.session_state.vectordb)
            st.markdown("### Response")
            st.code(answer, language="markdown")
        except Exception as e:
            st.error(f"Error: {e}")
