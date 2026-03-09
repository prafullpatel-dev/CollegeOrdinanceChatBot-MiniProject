# ──────────────────────────────────────────────────────────────────────────────
#  rag_backend.py  —  Pure LangChain RAG pipeline (no Streamlit here)
# ──────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ─── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
RETRIEVER_K     = 3

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question based on the context "
    "provided below. If the context is insufficient, just say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Conversation History:\n{history}\n\n"
    "Question: {question}"
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_docs(retrieved_docs: list) -> str:
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


def _format_history(messages: list) -> str:
    """Turn the messages list into a readable conversation log."""
    if not messages:
        return "No previous conversation."
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ─── Core pipeline ────────────────────────────────────────────────────────────

def load_and_split_pdfs(file_paths: list) -> list:
    splitter   = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        chunks = splitter.split_documents(loader.lazy_load())
        all_chunks.extend(chunks)
    return all_chunks


def build_vector_store(chunks: list) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def build_retriever(vector_store: FAISS):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


def build_rag_chain(retriever):
    """
    Build a chain that accepts a dict: {"question": str, "history": str}

    Flow:
        {"question", "history"}
            → RunnableParallel: retrieve context / pass question / pass history
            → PromptTemplate (fills context + history + question)
            → LLM
            → StrOutputParser
    """
    llm    = ChatGroq(model=LLM_MODEL)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "history", "question"],
    )
    parser = StrOutputParser()

    # The chain now accepts a dict input — extract each field explicitly
    parallel_chain = RunnableParallel({
        "context" : (lambda x: x["question"]) | retriever | RunnableLambda(_format_docs),
        "question": lambda x: x["question"],
        "history" : lambda x: x["history"],
    })

    return parallel_chain | prompt | llm | parser


def process_pdfs(file_paths: list):
    """Load PDFs → embed → return a ready-to-use chain."""
    chunks       = load_and_split_pdfs(file_paths)
    vector_store = build_vector_store(chunks)
    retriever    = build_retriever(vector_store)
    return build_rag_chain(retriever)


def ask(chain, question: str, messages: list = None) -> str:
    """
    Invoke the RAG chain with the current question and full chat history.

    Args:
        chain    : returned by process_pdfs()
        question : current user question (not yet in messages)
        messages : list of {"role": "user"|"assistant", "content": str}
                   representing all PRIOR turns

    Returns:
        Answer string.
    """
    history_text = _format_history(messages or [])

    # Single clean invoke — history travels with the question as a dict
    return chain.invoke({
        "question": question,
        "history" : history_text,
    })