import os, re, requests
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
# import streamlit as st

def load_documents():
    docs = []
    for file in os.listdir("docs"):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join("docs", file))
            docs.extend(loader.load())
    return docs

def create_vectorstore(docs):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name="rag-chain"
    )
    return vectordb

flan_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=flan_pipe)

# try:
#     hf_token = st.secrets["HUGGINGFACE_TOKEN"]
# except KeyError:
#     st.error("Hugging Face token not found in Streamlit Secrets. Please configure it.")
#     st.stop()

# try:
#     mistral_pipeline = pipeline(
#         "text-generation",
#         model="mistralai/Mistral-7B-v0.1",
#         max_new_tokens=256,
#         token=hf_token,
#     )
#     llm = HuggingFacePipeline(pipeline=mistral_pipeline)
# except Exception as e:
#     st.error(f"Error initializing Mistral model: {e}")
#     st.stop()


prompt = PromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question: {question}
""")

def calculate(query: str) -> str:
    try:
        expr = re.findall(r"[-+*/().0-9\s]+", query)[0]
        return str(eval(expr))
    except:
        return "Could not evaluate the expression."

def define(query: str) -> str:
    try:
        word = query.split()[-1]
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        res = requests.get(url).json()
        return res[0]['meanings'][0]['definitions'][0]['definition']
    except:
        return f"Definition not found for {word}."

def retrieve_context(query, vectordb):
    results = vectordb.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

def rag_chain(question: str, vectordb) -> str:
    context = retrieve_context(question, vectordb)
    final_prompt = prompt.invoke({"context": context, "question": question})
    return llm.invoke(final_prompt)

def route_question(question: str, vectordb) -> str:
    q = question.lower()
    if any(k in q for k in ["add", "subtract", "multiply", "divide", "+", "-", "*", "/", "calculate"]):
        return "Calculator Tool\n\n" + calculate(question)
    elif any(k in q for k in ["define", "meaning", "definition"]):
        return "Dictionary Tool\n\n" + define(question)
    else:
        return "RAG Tool\n\n" + rag_chain(question, vectordb)
