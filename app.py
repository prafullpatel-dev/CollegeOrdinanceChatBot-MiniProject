import streamlit as st
import tempfile
import os
from pathlib import Path
from rag_backend import process_pdfs, ask


# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="DMPC Ordinance Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Session State Initialization ─────────────────────────────────────────────
# Initialize BEFORE using any session state variables

if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

if "uploaded_files_count" not in st.session_state:
    st.session_state.uploaded_files_count = 0


# ─── Custom CSS (Dark Mode) ──────────────────────────────────────────────────

st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #64b5f6;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .user-message {
        background-color: #1e3a5f;
        border-left: 4px solid #4a9eff;
    }
    .assistant-message {
        background-color: #2d2d2d;
        border-left: 4px solid #66bb6a;
    }
    .message-role {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .message-content {
        line-height: 1.6;
        color: #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .upload-section {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px dashed #444;
        margin-bottom: 1rem;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #242424;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    /* Info boxes */
    .stAlert {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def save_uploaded_files(uploaded_files):
    """Save uploaded PDFs to temp directory and return paths."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths


def process_uploaded_pdfs(uploaded_files):
    """Process PDFs and initialize the RAG chain."""
    with st.spinner("🔄 Processing PDFs... This may take a moment."):
        file_paths = save_uploaded_files(uploaded_files)
        st.session_state.chain = process_pdfs(file_paths)
        st.session_state.pdfs_processed = True
        st.session_state.uploaded_files_count = len(uploaded_files)
    st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s)!")


def clear_conversation():
    """Clear the conversation history."""
    st.session_state.messages = []
    st.rerun()


def reset_all():
    """Reset everything including uploaded PDFs."""
    st.session_state.chain = None
    st.session_state.messages = []
    st.session_state.pdfs_processed = False
    st.session_state.uploaded_files_count = 0
    st.rerun()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload DMPC Ordinance PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to analyze",
        key="pdf_uploader"
    )
    
    if uploaded_files:
        if len(uploaded_files) != st.session_state.uploaded_files_count:
            if st.button("🚀 Process PDFs", type="primary"):
                process_uploaded_pdfs(uploaded_files)
    
    st.markdown("---")
    
    # Status indicator
    if st.session_state.pdfs_processed:
        st.success(f"✓ {st.session_state.uploaded_files_count} PDF(s) loaded")
    else:
        st.info("📤 Please upload PDF files to begin")
    
    st.markdown("---")
    
    # Action buttons
    st.markdown("### ⚙️ Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", disabled=not st.session_state.messages):
            clear_conversation()
    
    with col2:
        if st.button("🔄 Reset All", disabled=not st.session_state.pdfs_processed):
            reset_all()
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Upload** one or more PDF files
    2. Click **Process PDFs** button
    3. **Ask questions** about the documents
    4. View **answers** based on document content
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This chatbot uses **RAG (Retrieval-Augmented Generation)** 
    to answer questions from your DMPC ordinance documents.
    
    - **Model:** Llama 3.3 70B
    - **Embeddings:** HuggingFace MiniLM
    - **Vector Store:** FAISS
    """)


# ─── Main Content ─────────────────────────────────────────────────────────────

# Header
st.markdown('<div class="main-header">📚 DMPC Ordinance Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your uploaded ordinance documents</div>', unsafe_allow_html=True)

# Main chat area
if not st.session_state.pdfs_processed:
    st.info("👈 Please upload and process PDF files from the sidebar to get started!")
else:
    # Display chat history
    if st.session_state.messages:
        st.markdown("### 💬 Conversation History")
        
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-role" style="color: #4a9eff;">👤 You</div>
                    <div class="message-content">{content}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-role" style="color: #66bb6a;">🤖 Assistant</div>
                    <div class="message-content">{content}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("### 💬 Start a Conversation")
        st.info("👇 Type your question below to get started!")
    
    # Chat input
    st.markdown("---")
    question = st.chat_input("Ask a question about your documents...", key="chat_input")
    
    if question:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # Get response from RAG chain
        with st.spinner("🤔 Thinking..."):
            try:
                # Pass all messages except the current one as history
                history = st.session_state.messages[:-1]
                response = ask(
                    chain=st.session_state.chain,
                    question=question,
                    messages=history
                )
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Rerun to display the new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                # Remove the user message if there was an error
                st.session_state.messages.pop()