# 🧠 Autism Research Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based exclusively on a curated collection of autism research articles. The system ensures responses are grounded in the provided documents, preventing hallucinations and unsupported claims.

## ✨ Features

- **Document-grounded responses** — Only answers from your provided research articles
- **Source citations** — Every answer includes which documents were used
- **Semantic search** — Finds relevant content using embedding-based similarity
- **Web interface** — Clean Gradio UI for easy interaction
- **Persistent storage** — ChromaDB stores embeddings between sessions

## 🏗️ Architecture

```
┌─────────────────┐
│  articles/*.txt │  ← Your research documents
└────────┬────────┘
         │ 1. Chunk & embed
         ▼
┌─────────────────┐
│ SentenceTransf. │  ← Local embedding model (free)
│ all-MiniLM-L6   │
└────────┬────────┘
         │ 2. Store vectors
         ▼
┌─────────────────┐
│   ChromaDB      │  ← Vector database
└────────┬────────┘
         │ 3. Similarity search
         ▼
┌─────────────────┐     ┌─────────────────┐
│  User Question  │────►│  Gemini Flash   │────► Grounded Answer
└─────────────────┘     │  + Context      │
                        └─────────────────┘
```

## 📚 Research Source

The research articles used in this project come from the **Centre for Research in Autism and Education (CRAE)** at UCL.

> *"CRAE is a team of autistic and non-autistic people, headed by Dr Anna Remington. We conduct ground-breaking scientific research to enhance our knowledge about support, education and outcomes for autistic people, their families and those who support them."*

This is an amazing and inspiring team. Don't hesitate to have a look at what they do!

🔗 **[CRAE Academic Publications](https://crae.ioe.ac.uk/academic-publications/)**

*Note: The research papers could be publicly shared in this repository. Please visit the CRAE website to access the original publications.*

## 📋 Prerequisites

- Python 3.9+
- Google AI Studio API key ([Get one free here](https://aistudio.google.com/apikey))

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/autism-research-chatbot.git
   cd autism-research-chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```

5. **Add your research articles**
   
   Create an `articles/` folder and add your `.txt` files:
   ```bash
   mkdir articles
   # Add your .txt research files to this folder
   ```

## 📖 Usage

### Web Interface (Recommended)
```bash
python app.py
```
Then open `http://localhost:7860` in your browser.

### Command Line
```bash
python rag_system.py
```

## 📁 Project Structure

```
autism-research-chatbot/
├── articles/           # Your research documents (not tracked by git)
├── chroma_db/          # Vector database storage (auto-generated)
├── app.py              # Gradio web interface
├── rag_system.py       # Core RAG logic
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment variables
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## ⚙️ Configuration

You can adjust these parameters in `rag_system.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K` | 3 | Number of chunks to retrieve |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |

## 🔧 Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Web UI**: Gradio
- **Language**: Python 3.11

## 📝 How It Works

1. **Indexing**: Documents are split into overlapping chunks, converted to embeddings, and stored in ChromaDB
2. **Retrieval**: User questions are embedded and matched against stored chunks using cosine similarity
3. **Generation**: Retrieved chunks are passed to Gemini with instructions to only use the provided context
4. **Citation**: The system returns both the answer and the source documents used


## 🙏 Acknowledgments

- Built as part of an AI development course
- Uses open-source embedding models from Hugging Face
- Powered by Google's Gemini API
