from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time

# Import our custom modules
from rag.data_processor import DataProcessor
from rag.vector_store import VectorStore
from rag.generator import ResponseGenerator

app = FastAPI(title="HackACure RAG API", 
              description="Retrieval-Augmented Generation API for medical textbooks")

# Initialize components
data_processor = DataProcessor()
vector_store = VectorStore()
response_generator = ResponseGenerator()

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query and return an answer with supporting contexts"""
    start_time = time.time()
    
    try:
        # Validate input
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.top_k < 1 or request.top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
        
        # Retrieve relevant contexts
        contexts = vector_store.retrieve(request.query, request.top_k)
        
        # Generate response
        answer = response_generator.generate(request.query, contexts)
        
        # Ensure we're within time limit (60 seconds)
        elapsed_time = time.time() - start_time
        if elapsed_time > 55:  # Leave some buffer
            print(f"Warning: Query took {elapsed_time:.2f} seconds, approaching timeout")
        
        return QueryResponse(answer=answer, contexts=contexts)
    
    except Exception as e:
        # Log the error but return a generic message to the user
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred processing your query")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)