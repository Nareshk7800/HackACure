import os
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class VectorStore:
    """Manages vector embeddings and semantic search functionality"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "index"):
        """Initialize the vector store
        
        Args:
            model_name: Name of the sentence transformer model to use
            index_path: Directory to store the FAISS index
        """
        self.model_name = model_name
        self.index_path = index_path
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = None
        self.chunks = []
        
        # Create index directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.model.encode(text)
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build a FAISS index from text chunks
        
        Args:
            chunks: List of text chunks with metadata
        """
        self.chunks = chunks
        
        # Create a new index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Get embeddings for all chunks
        embeddings = []
        for chunk in tqdm(chunks, desc="Creating embeddings"):
            embedding = self._get_embedding(chunk["chunk_text"])
            embeddings.append(embedding)
        
        # Convert to numpy array and add to index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        print(f"Built index with {len(chunks)} chunks")
        
        # Save the index
        self._save_index()
    
    def _save_index(self):
        """Save the FAISS index to disk"""
        if self.index is not None:
            index_file = os.path.join(self.index_path, "faiss.index")
            faiss.write_index(self.index, index_file)
            print(f"Saved index to {index_file}")
    
    def _load_index(self):
        """Load the FAISS index from disk"""
        index_file = os.path.join(self.index_path, "faiss.index")
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            print(f"Loaded index from {index_file}")
            return True
        return False
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve the most relevant chunks for a query
        
        Args:
            query: The query text
            top_k: Number of results to return
            
        Returns:
            List of relevant text chunks
        """
        # Load index if not already loaded
        if self.index is None:
            if not self._load_index():
                raise ValueError("No index available. Please build the index first.")
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get the corresponding chunks
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                # Get the chunk text, limiting to 3 lines max
                chunk_text = self.chunks[idx]["chunk_text"]
                lines = chunk_text.split('\n')
                if len(lines) > 3:
                    # Take the most relevant 3 lines (middle of the chunk)
                    middle_idx = len(lines) // 2
                    snippet = '\n'.join(lines[max(0, middle_idx-1):min(len(lines), middle_idx+2)])
                else:
                    snippet = chunk_text
                
                results.append(snippet)
        
        return results