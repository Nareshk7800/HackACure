import os
import sys
from dotenv import load_dotenv
from rag.data_processor import DataProcessor
from rag.vector_store import VectorStore

# Load environment variables
load_dotenv()

def main():
    """Index all textbooks and build the vector store"""
    print("Starting textbook indexing process...")
    
    # Initialize components
    data_processor = DataProcessor()
    vector_store = VectorStore()
    
    # Process textbooks into chunks
    print("Processing textbooks into chunks...")
    chunks = data_processor.get_chunks_for_indexing()
    
    if not chunks:
        print("Error: No chunks were generated. Check the textbook files.")
        sys.exit(1)
    
    # Build the vector index
    print(f"Building vector index with {len(chunks)} chunks...")
    vector_store.build_index(chunks)
    
    print("Indexing complete! The RAG API is ready to use.")

if __name__ == "__main__":
    main()