import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

def load_embeddings(embeddings_dir: str = 'embeddings') -> List[Dict[str, Any]]:
    """Load all embedding files from the embeddings directory."""
    all_embeddings = []
    for file in Path(embeddings_dir).glob('*_embeddings.json'):
        with open(file, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
            all_embeddings.extend(embeddings)
    return all_embeddings

def create_vector_store(embeddings_data: List[Dict[str, Any]]) -> FAISS:
    """Create a FAISS vector store from the embeddings data."""
    texts = [item['text'] for item in embeddings_data]
    embeddings = [item['embedding'] for item in embeddings_data]
    
    index = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=HuggingFaceEmbeddings(),
        metadatas=[{'chunk_id': item['chunk_id']} for item in embeddings_data]
    )
    
    return index

def create_rag_chain(vector_store: FAISS):
    """Create a RAG chain using the vector store."""
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create prompt template
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer: Let me help you with that."""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=genai.GenerativeModel('gemini-pro'),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain

def ask_question(chain, question: str):
    """Ask a question to the RAG system."""
    result = chain({"query": question})
    
    print("\nQuestion:", question)
    print("\nAnswer:", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"- {doc.page_content[:200]}...")

def main():
    # Load embeddings
    print("Loading embeddings...")
    embeddings_data = load_embeddings()
    print(f"Loaded {len(embeddings_data)} chunks with embeddings")
    
    # Create vector store
    print("\nCreating vector store...")
    vector_store = create_vector_store(embeddings_data)
    
    # Create RAG chain
    print("\nCreating RAG chain...")
    rag_chain = create_rag_chain(vector_store)
    
    # Test questions
    test_questions = [
        "What are the main topics discussed in the documents?",
        "Can you summarize the key points from the documents?",
    ]
    
    print("\nTesting with sample questions:")
    for question in test_questions:
        print("\n" + "="*80)
        ask_question(rag_chain, question)
    
    # Interactive Q&A
    print("\nStarting interactive Q&A session (type 'quit' to exit):")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
        ask_question(rag_chain, question)

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables")
    else:
        main()
