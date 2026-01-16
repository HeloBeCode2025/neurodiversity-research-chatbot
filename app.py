# app.py
import gradio as gr
from rag_system import load_and_index_docs, ask, collection

# Initialize on startup
load_and_index_docs()

def chat(message: str, history: list) -> str:
    """Handle chat messages."""
    if not message.strip():
        return "Please ask a question!"
    
    result = ask(message)
    
    # Format response with sources
    sources_text = "\n".join([
        f"- {s['source']} ({s['relevance']} relevant)" 
        for s in result['sources']
    ])
    
    return f"{result['answer']}\n\n---\n📚 **Sources:**\n{sources_text}"


# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="∞ Autism Research Chat ∞",
    description=f"Ask questions about the {collection.count()} indexed document chunks. I'll only answer from what's in the documents!",
    examples=[
        "What is autism?",
        "Why is it important to include autistic people in research about autism?",
        "How can we support autistic job seekers?"
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)  # Set share=True for public link