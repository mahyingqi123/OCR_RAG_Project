import os
import PyPDF2
from google import genai
from pathlib import Path
from dotenv import load_dotenv
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def generate_embedding(text: str) -> list:
    """Generate embedding using Gemini API."""
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return result.embeddings[0].values

def process_pdfs_to_embeddings(input_dir='processed_pdfs', output_dir='embeddings'):
    """Process PDFs and generate embeddings."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    # Get all processed PDF files
    pdf_files = list(Path(input_dir).glob('processed_*.pdf'))
    
    if not pdf_files:
        print(f"No processed PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} processed PDF files")
    
    # Process each PDF
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(str(pdf_path))
            
            # Split text into chunks using LangChain
            chunks = text_splitter.split_text(text)
            print(f"Split into {len(chunks)} chunks")
            
            # Generate embeddings for each chunk
            embeddings_data = []
            for i, chunk in enumerate(chunks):
                print(f"Generating embedding for chunk {i+1}/{len(chunks)}")
                embedding = generate_embedding(chunk)
                embeddings_data.append({
                    'chunk_id': i,
                    'text': chunk,
                    'embedding': embedding
                })
            
            # Save embeddings to JSON file
            output_file = Path(output_dir) / f"{pdf_path.stem}_embeddings.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
            
            print(f"Successfully processed {pdf_path.name}")
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables")
    else:
        process_pdfs_to_embeddings() 