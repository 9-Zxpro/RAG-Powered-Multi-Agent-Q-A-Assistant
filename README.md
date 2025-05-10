# RAG-Powered-Multi-Agent-Q-A-Assistant

## This Rag powered multi agent assistant uses google/flan-t5-base model

I have created three documents in docs folder for prepareing document corpus. Created a vectorstore using Chroma.
I haven't used Langchain's Tool to route, but used function to simulate the same.
If any query is passed having word like "define", "meaning", "definition", it will route to dictionary.
If having word like "add", "subtract", "multiply", "divide", "+", "-", "*", "/", "calculate" will route to calulator.
Otherwise rag_chain.

## Features

- Retrieves relevant chunks from a small document corpus using Chroma
- Uses Hugging Face LLM (Flan-T5) for answer generation
- Routes queries to:
  - Calculator (for math)
  - Dictionary (for definitions)
  - RAG → LLM (for factual/document-based queries)
- Streamlit UI

---

## Project Structure
- docs
  - company_info.txt
  - faq.txt
  - product_specs.txt
- agent.py
- app.py
- requirements.txt