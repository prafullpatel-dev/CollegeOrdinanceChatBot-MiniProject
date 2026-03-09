# DMPC Ordinance RAG Chatbot 📚

A Streamlit-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from DMPC ordinance PDF documents.

## Features ✨

- **Multiple PDF Upload**: Upload one or more ordinance PDFs simultaneously
- **Conversational AI**: Natural language Q&A with chat history
- **RAG Pipeline**: Combines document retrieval with LLM generation
- **Clean UI**: Professional interface designed for college students
- **Session Management**: Clear chat history or reset entire session

## Tech Stack 🛠️

- **Frontend**: Streamlit
- **LLM**: Llama 3.3 70B (via Groq)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Framework**: LangChain

## Project Structure 📁

```
.
├── app.py              # Streamlit frontend
├── rag_backend.py      # LangChain RAG pipeline
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── README.md          # This file
```

## Setup Instructions 🚀

### 1. Clone or Download the Project

```bash
git clone <your-repo-url>
cd <project-directory>
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

**Get your Groq API key:**

1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste it into your `.env` file

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## How to Use 📖

1. **Upload PDFs**: Click the file uploader in the sidebar and select one or more PDF files
2. **Process**: Click the "🚀 Process PDFs" button
3. **Ask Questions**: Type your question in the chat input at the bottom
4. **View Answers**: See AI-generated responses based on your documents
5. **Continue Chatting**: Ask follow-up questions - the bot remembers context
6. **Clear/Reset**: Use sidebar buttons to clear chat or reset everything

## Configuration ⚙️

You can adjust RAG parameters in `rag_backend.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 1000      # Size of text chunks
CHUNK_OVERLAP   = 200       # Overlap between chunks
RETRIEVER_K     = 3         # Number of chunks to retrieve
```

## Troubleshooting 🔧

### Common Issues

**Issue**: "No module named 'streamlit'"

- **Solution**: Make sure you've activated your virtual environment and installed requirements

**Issue**: "GROQ_API_KEY not found"

- **Solution**: Verify your `.env` file exists and contains `GROQ_API_KEY=...`

**Issue**: PDFs not processing

- **Solution**: Ensure PDFs are text-based (not scanned images). For scanned PDFs, OCR preprocessing is needed.

**Issue**: Slow response times

- **Solution**: Groq API is generally fast, but large PDFs may take time to process initially

## Architecture 🏗️

```
┌─────────────────┐
│  Upload PDFs    │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  PyPDFLoader        │
│  (Load & Split)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  HuggingFace        │
│  Embeddings         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  FAISS Vector Store │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Retriever          │
│  (K=3 chunks)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  User Question      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Groq LLM           │
│  (Llama 3.3 70B)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Answer             │
└─────────────────────┘
```

## Credits 👥

Created as a mini project for college students to interact with DMPC ordinance documents.

## License 📄

MIT License - Feel free to use and modify for your projects!

## Support 💬

For issues or questions, please create an issue in the repository or contact your project supervisor.

---

**Happy Chatting!** 🎉
