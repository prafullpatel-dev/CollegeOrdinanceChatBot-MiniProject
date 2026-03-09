# ──────────────────────────────────────────────────────────────────────────────
#  app.py  —  Streamlit UI  (imports RAG logic from rag_backend.py)
# ──────────────────────────────────────────────────────────────────────────────

import tempfile
import streamlit as st
from rag_backend import process_pdfs, ask


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — PDF Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

  :root {
    --bg:       #0d0f14;
    --surface:  #161921;
    --border:   #252836;
    --accent:   #e8ff3c;
    --accent2:  #3cffe8;
    --muted:    #4a4f62;
    --text:     #e8ecf4;
    --text-dim: #8892a4;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
  }
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  #MainMenu, footer, header { visibility: hidden; }

  .docmind-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.04em;
    line-height: 1.1;
    color: var(--text);
    margin-bottom: 0.15rem;
  }
  .docmind-title span { color: var(--accent); }
  .docmind-sub {
    font-size: 0.78rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }
  .sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.4rem 0 0.5rem;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
  }
  .status-ready { background: rgba(232,255,60,.12); color: var(--accent);  border: 1px solid rgba(232,255,60,.3); }
  .status-idle  { background: rgba(74,79,98,.15);   color: var(--text-dim);border: 1px solid var(--border);       }

  .chat-user      { display:flex; justify-content:flex-end;   margin:0.6rem 0; }
  .chat-assistant { display:flex; justify-content:flex-start;  margin:0.6rem 0; }
  .chat-user .bubble {
    background: var(--accent);
    color: #0d0f14;
    border-radius: 18px 18px 4px 18px;
    padding: 0.65rem 1rem;
    max-width: 72%;
    font-size: 0.88rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    line-height: 1.55;
  }
  .chat-assistant .bubble {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 0.65rem 1rem;
    max-width: 78%;
    font-size: 0.88rem;
    font-family: 'DM Mono', monospace;
    line-height: 1.65;
  }

  [data-testid="stTextInput"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1rem !important;
  }
  [data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,255,60,.15) !important;
  }

  .stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
  }
  .stButton > button:first-child {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
  }
  .stButton > button:first-child:hover {
    background: #d4eb2e !important;
    transform: translateY(-1px);
  }
  .stop-btn .stButton > button {
    background: transparent !important;
    color: #ff6b6b !important;
    border: 1px solid rgba(255,107,107,0.35) !important;
  }
  .stop-btn .stButton > button:hover { background: rgba(255,107,107,.08) !important; }

  [data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
  }
  .pdf-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(60,255,232,.06);
    border: 1px solid rgba(60,255,232,.2);
    color: var(--accent2);
    border-radius: 6px; padding: 3px 9px;
    font-size: 0.7rem; margin: 2px 3px;
    font-family: 'DM Mono', monospace;
  }
  .history-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(232,255,60,.06);
    border: 1px solid rgba(232,255,60,.2);
    color: var(--accent);
    border-radius: 6px; padding: 3px 9px;
    font-size: 0.68rem; margin-top: 6px;
    font-family: 'DM Mono', monospace;
  }
  .empty-state {
    text-align: center; padding: 4rem 2rem;
    color: var(--muted); font-size: 0.82rem; line-height: 2;
  }
  .empty-icon { font-size: 3rem; margin-bottom: 0.5rem; }
  hr { border-color: var(--border) !important; }
  .chat-scroll {
    max-height: 58vh; overflow-y: auto; padding-right: 6px;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages"       : [],   # [{"role": "user"|"assistant", "content": "..."}]
    "rag_chain"      : None,
    "chat_stopped"   : False,
    "processed_files": [],
    "input_key"      : 0,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown('<div class="docmind-title">Doc<span>Mind</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="docmind-sub">PDF Intelligence Layer</div>',  unsafe_allow_html=True)

    # Status pill
    if st.session_state.rag_chain:
        st.markdown('<span class="status-pill status-ready">● Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-idle">○ Awaiting PDFs</span>', unsafe_allow_html=True)

    # History counter badge
    turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    if turn_count:
        st.markdown(
            f'<span class="history-badge">🕘 {turn_count} turn{"s" if turn_count != 1 else ""} in context</span>',
            unsafe_allow_html=True,
        )

    # Upload
    st.markdown('<div class="sidebar-label">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.markdown("**Loaded files:**")
        for f in uploaded_files:
            st.markdown(f'<span class="pdf-badge">📄 {f.name}</span>', unsafe_allow_html=True)

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚡ Process", use_container_width=True)
    with col2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_btn = st.button("⏹ Stop Chat", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🗑  Clear History", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.chat_stopped = False
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<span style="font-size:0.68rem;color:#4a4f62;">'
        'Powered by LangChain · Groq · FAISS</span>',
        unsafe_allow_html=True,
    )


# ─── Process PDFs ─────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("🔍 Embedding documents…"):
            tmp_paths = []
            for uf in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                    tmp.write(uf.read())
                    tmp_paths.append(tmp.name)

            st.session_state.rag_chain       = process_pdfs(tmp_paths)
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.session_state.messages        = []
            st.session_state.chat_stopped    = False

        st.sidebar.success(f"✓ {len(uploaded_files)} PDF(s) indexed!")
        st.rerun()

if stop_btn:
    st.session_state.chat_stopped = True
    st.rerun()


# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.5rem;'
    'letter-spacing:-0.03em;margin-bottom:0.1rem;">Chat with your documents</h2>',
    unsafe_allow_html=True,
)

if st.session_state.processed_files:
    badges = "".join(
        f'<span class="pdf-badge">📄 {n}</span>'
        for n in st.session_state.processed_files
    )
    st.markdown(badges, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ─── Chat History Display ─────────────────────────────────────────────────────
with st.container():
    if not st.session_state.messages:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">🧠</div>'
            'Upload and process your PDFs,<br>then ask anything about them.'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><div class="bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-assistant"><div class="bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Input Row ─────────────────────────────────────────────────────────────────
if not st.session_state.chat_stopped:

    col_input, col_send = st.columns([8, 1])
    with col_input:
        user_input = st.text_input(
            "question",
            placeholder="Ask a question about your documents…",
            label_visibility="collapsed",
            key=f"user_input_{st.session_state.input_key}",
        )
    with col_send:
        send_btn = st.button("Send →", use_container_width=True)

    if (send_btn or user_input) and user_input.strip():
        if not st.session_state.rag_chain:
            st.warning("⚠️ Please upload and process your PDFs first.")
        else:
            question = user_input.strip()

            # Pass the current history BEFORE appending the new question,
            # so the LLM sees only completed prior turns as context.
            prior_messages = list(st.session_state.messages)

            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("Thinking…"):
                try:
                    # ── history-aware call into the backend ───────────────────
                    answer = ask(
                        st.session_state.rag_chain,
                        question,
                        messages=prior_messages,   # ← full prior conversation
                    )
                except Exception as e:
                    answer = f"⚠️ Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.input_key += 1
            st.rerun()

else:
    st.markdown(
        '<div style="text-align:center;padding:1.5rem;color:#ff6b6b;'
        'border:1px solid rgba(255,107,107,.25);border-radius:10px;'
        'font-size:0.82rem;margin-top:1rem;">'
        "⏹ Chat stopped. Click <b>Clear History</b> in the sidebar to restart."
        "</div>",
        unsafe_allow_html=True,
    )