# PDF RAG Chatbot 📚🤖

A local Retrieval-Augmented Generation (RAG) system for intelligent PDF question answering using LangChain, FAISS, and Ollama.

## Features 

- **PDF Document Processing** - Upload and query any PDF document
- **Semantic Search** - FAISS vector database for fast retrieval
- **Conversation Memory** - Maintains context across multiple questions
- **Smart Retrieval** - Choose between MMR (diverse results) or similarity search
- **Page Citations** - Answers include source page references
- **100% Local** - No API keys, runs entirely on your machine

## Prerequisites 

- Python 3.8+
- [Ollama](https://ollama.com/download/OllamaSetup.exe) installed and running
- 8GB+ RAM recommended

## Installation 🚀

### 1. Clone the Repository
```bash
git clone https://github.com/AkhilKandoi/Local-PDF-RAG.git
cd Local-PDF-RAG
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama Models
Make sure Ollama is running, then pull the required models:

```bash
# Pull the LLM (Large Language Model)
ollama pull llama3.1:8b

# Pull the embedding model
ollama pull nomic-embed-text:latest
```

**Verify models are installed:**
```bash
ollama list
```

You should see both `llama3.1:8b` and `nomic-embed-text:latest`.

## Usage 

### Start the Application
```bash
streamlit run PDF_RAG.py
```

The app will open automatically in your browser.

### Using the Chatbot

1. **Upload PDF**: Click "Upload a PDF" in the sidebar
2. **Configure Settings** (optional):
   - **Retrieval Strategy**: 
     - `MMR` - More diverse results (recommended)
     - `Similarity` - Most relevant results
   - **Top-k Chunks**: Number of document chunks to retrieve (3-10)
3. **Ask Questions**: Type your question in the chat input
4. **View Answers**: Get responses with page citations

### Example Questions

Try these with a technical paper or textbook:

```
- "What is the main topic of this document?"
- "Explain the methodology used on page 5"
- "Summarize the key findings"
- "What are the limitations mentioned?"
- "Compare the results in section 3 and section 4"
```

## Project Structure 📁

```
Local-PDF-RAG/
├── PDF_RAG.py           # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── temp.pdf            # Temporary file (auto-generated)
```

## How It Works 🔧

1. **Document Loading**: PyMuPDF extracts text from uploaded PDF
2. **Text Splitting**: Recursive splitter creates 800-character chunks with 200-char overlap
3. **Embedding**: Nomic embeddings convert chunks to vectors
4. **Vector Storage**: FAISS stores embeddings for fast retrieval
5. **Retrieval**: User query → similar chunks retrieved (MMR or similarity)
6. **Generation**: Llama 3.1 generates answer using retrieved context + chat history
7. **Memory**: Conversation stored in session for follow-up questions

## Configuration ⚙️

### Modify Chunk Size
Edit `PDF_RAG.py` line 75:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Increase for more context per chunk
    chunk_overlap=200    # Increase for better continuity
)
```

### Change LLM Model
Edit `PDF_RAG.py` line 92:
```python
llm = ChatOllama(
    model="llama3.1:8b"  # Try: "llama3.2:3b", "mistral:7b", etc.
)
```

### Adjust Retrieval Parameters
Edit `PDF_RAG.py` lines 87-88:
```python
# For MMR
search_kwargs={'k': 5, "fetch_k": 20, "lambda_mult": 0.7}
# k = final chunks, fetch_k = candidates, lambda_mult = diversity (0=diverse, 1=similar)
```

## Requirements 📦

```txt
langchain-community
langchain-ollama
langchain-core
langchain-text-splitters
faiss-cpu
pymupdf
streamlit
```

## Limitations ⚠️

- Text-only (images/tables not processed)
- No OCR for scanned PDFs
- Context limited to ~6 previous messages
- Local models less powerful than GPT-4/Claude/Gemini

## Future Improvements 🚀

- Multi-PDF support
- Table extraction and parsing
- Evaluation metrics
- Source highlighting in PDF viewer
- Query suggestions

