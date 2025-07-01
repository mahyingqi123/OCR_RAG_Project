import os
import ocrmypdf
from pathlib import Path

def process_pdfs(input_dir='pdfs', output_dir='processed_pdfs'):
    """
    Process all PDFs in the input directory using OCRmyPDF and save to output directory.
    
    Args:
        input_dir (str): Directory containing input PDFs
        output_dir (str): Directory to save processed PDFs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files in the input directory
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_path in pdf_files:
        output_path = Path(output_dir) / f"processed_{pdf_path.name}"
        print(f"Processing {pdf_path.name}...")
        
        try:
            # Process the PDF with OCRmyPDF
            ocrmypdf.ocr(
                input_file=str(pdf_path),
                output_file=str(output_path),
            )
            print(f"Successfully processed {pdf_path.name}")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")

if __name__ == "__main__":
    process_pdfs() 