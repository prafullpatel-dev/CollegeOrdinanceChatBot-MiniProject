# ──────────────────────────────────────────────────────────────────────────────
#  rag_backend.py  —  Pure LangChain RAG pipeline (no Streamlit here)
# ──────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ─── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
RETRIEVER_K     = 3


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_docs(retrieved_docs: list) -> str:
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


def _build_chat_messages(context: str, history: list, question: str) -> list:
    """
    Construct the full message list for ChatGroq:

        SystemMessage  — persona + context (PDF chunks)
        HumanMessage   ─┐
        AIMessage      ─┤  ← prior turns from history
        HumanMessage   ─┘
        HumanMessage   — current question  (always last)
    """
    messages = [
        SystemMessage(content=(
            "You are a knowledgeable and conversational assistant. "
            "Answer questions using ONLY the document context provided below. "
            "If the context does not contain enough information, say you don't know. "
            "Be concise, clear, and natural — like a helpful colleague, not a search engine.\n\n"
            f"Document Context:\n{context}"
        ))
    ]

    # Replay prior turns as proper chat roles
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Current question always goes last as a HumanMessage
    messages.append(HumanMessage(content=question))

    return messages


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
    Chain input : {"question": str, "history": list}
    Chain output: answer string

    Flow:
        RunnableParallel → retrieves context, passes question & history through
        RunnableLambda   → assembles proper ChatPromptTemplate message list
        ChatGroq         → responds as a conversational agent
        StrOutputParser  → extracts plain string
    """
    llm    = ChatGroq(model=LLM_MODEL, temperature=0.3)
    parser = StrOutputParser()

    # Step 1 — retrieve context in parallel, pass other fields through
    parallel_chain = RunnableParallel({
        "context" : (lambda x: x["question"]) | retriever | RunnableLambda(_format_docs),
        "question": lambda x: x["question"],
        "history" : lambda x: x["history"],
    })

    # Step 2 — build the ChatPromptTemplate message list from the parallel output
    def assemble_messages(inputs: dict) -> list:
        return _build_chat_messages(
            context  = inputs["context"],
            history  = inputs["history"],
            question = inputs["question"],
        )

    message_builder = RunnableLambda(assemble_messages)

    return parallel_chain | message_builder | llm | parser


def process_pdfs(file_paths: list):
    """Load PDFs → embed → return a ready-to-use chain."""
    chunks       = load_and_split_pdfs(file_paths)
    vector_store = build_vector_store(chunks)
    retriever    = build_retriever(vector_store)
    return build_rag_chain(retriever)


def ask(chain, question: str, messages: list = None) -> str:
    """
    Invoke the RAG chain.

    Args:
        chain    : returned by process_pdfs()
        question : current user question (NOT yet in messages)
        messages : all prior turns as [{"role": "user"|"assistant", "content": str}]

    Returns:
        Answer string from ChatGroq.
    """
    return chain.invoke({
        "question": question,
        "history" : messages or [],   # raw list — _build_chat_messages handles formatting
    })