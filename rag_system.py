import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv
import time

ARTICLES_DIR = "articles"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded properly
if not API_KEY:
    raise Exception('GOOGLE_API_KEY not found. Please set it in your .env file.')

# Initialize the new Gemini client
client = genai.Client(api_key=API_KEY)

# Initialize embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Initialize database
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="articles",
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk)
        start = end - overlap  # FIX: moved outside if block to prevent infinite loop
    return chunks


def load_and_index_docs():
    """Load all articles and add to vector store."""
    articles_path = Path(ARTICLES_DIR)

    # Check if articles directory exists
    if not articles_path.exists():
        raise Exception(f"Articles directory '{ARTICLES_DIR}' not found. Please create it and add .txt files.")

    # Check if already indexed
    if collection.count() > 0:
        print(f"Database already has {collection.count()} chunks. Skipping indexing.")
        return

    all_chunks = []
    all_ids = []
    all_metadatas = []

    # FIX: Added .glob() method
    txt_files = list(articles_path.glob("*.txt"))
    
    if not txt_files:
        raise Exception(f"No .txt files found in '{ARTICLES_DIR}'. Please add some text files.")
    
    print(f"Found {len(txt_files)} text files to process.")

    for file_path in txt_files:
        print(f"Processing: {file_path.name}")
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{file_path.stem}_{i}")
            all_metadatas.append({
                "source": file_path.name,
                "chunk_index": i
            })

    # Check we have chunks before trying to embed
    if not all_chunks:
        raise Exception("No text chunks generated. Check that your .txt files contain text.")

    # Generate embeddings
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)

    # Add to ChromaDB
    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        ids=all_ids,
        metadatas=all_metadatas
    )
    print(f"Indexed {len(all_chunks)} chunks from {len(txt_files)} files.")


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve most relevant chunks for a query."""
    query_embedding = embedder.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # FIX: separate strings
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "relevance": 1 - dist  # distance --> similarity
        })

    return retrieved


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate answer using Gemini with retrieved context."""

    # Format context with sources
    context_text = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in context_chunks
    ])

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
1. Only use information from the context below to answer
2. If the answer is not in the context, say "I don't know based on the provided documents."
3. Cite which source(s) you used
4. Be concise and direct

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

    #3 tries to avoid overload error
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    print(f"⏳ API overloaded, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return "Sorry, the AI service is currently overloaded. Please try again in a moment."
            else:
                raise e


def ask(query: str) -> dict:
    """Main RAG pipeline: retrieve + generate."""

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query)

    # Step 2: Generate answer with context
    answer = generate_answer(query, chunks)

    return {
        "question": query,
        "answer": answer,
        "sources": [{"source": c['source'], "relevance": f"{c['relevance']:.2%}"} for c in chunks]
    }


# ============ CLI INTERFACE ============
if __name__ == "__main__":
    print("🔧 Loading and indexing documents...")
    load_and_index_docs()

    print("\n✅ RAG System Ready! Type 'quit' to exit.\n")

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if not query:
            time.sleep(1)  # 1 second between queries
            continue

        result = ask(query)
        print(f"\n📝 Answer: {result['answer']}")
        print(f"📚 Sources used: {[s['source'] for s in result['sources']]}\n")