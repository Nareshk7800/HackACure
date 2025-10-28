import os
import PyPDF2
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataProcessor:
    """Handles processing of textbook PDFs into chunks suitable for indexing"""
    
    def __init__(self, data_dir: str = "HackACure-Dataset/Dataset", 
                 chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize the data processor
        
        Args:
            data_dir: Directory containing the PDF textbooks
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.data_dir = os.path.abspath(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from each page
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, source_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata
        
        Args:
            text: Text to split into chunks
            source_metadata: Metadata about the source document
            
        Returns:
            List of chunks with metadata
        """
        chunks = self.text_splitter.create_documents([text])
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata = {
                **source_metadata,
                "chunk_id": i,
                "chunk_text": chunk.page_content
            }
        
        return chunks
    
    def process_all_textbooks(self) -> List[Dict[str, Any]]:
        """Process all textbooks in the data directory
        
        Returns:
            List of all chunks with metadata
        """
        all_chunks = []
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
        
        for pdf_file in tqdm(pdf_files, desc="Processing textbooks"):
            pdf_path = os.path.join(self.data_dir, pdf_file)
            
            # Extract category from filename
            category = os.path.splitext(pdf_file)[0]
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            if text:
                # Create metadata
                metadata = {
                    "source": pdf_file,
                    "category": category,
                }
                
                # Chunk the text
                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)
        
        print(f"Processed {len(pdf_files)} textbooks into {len(all_chunks)} chunks")
        return all_chunks
    
    def get_chunks_for_indexing(self) -> List[Dict[str, Any]]:
        """Get all chunks ready for indexing
        
        Returns:
            List of chunks with text and metadata
        """
        return self.process_all_textbooks()