from typing import List, Dict, Any, Optional
import os
from openai import OpenAI

class ResponseGenerator:
    """Generates responses based on retrieved contexts"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize the response generator
        
        Args:
            model: The LLM model to use for response generation
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate(self, query: str, contexts: List[str]) -> str:
        """Generate a response based on the query and retrieved contexts
        
        Args:
            query: The user's query
            contexts: List of relevant context snippets
            
        Returns:
            Generated response
        """
        # Combine contexts into a single string
        context_text = "\n\n---\n\n".join(contexts)
        
        # Create the prompt
        prompt = f"""Answer the following medical question based ONLY on the provided context. 
        If the context doesn't contain enough information to answer the question fully, 
        state what you can answer based on the context and what information is missing. 
        Do not use any external knowledge. Keep your answer concise and factual.
        
        CONTEXT:
        {context_text}
        
        QUESTION: {query}
        
        ANSWER:"""
        
        # Generate response using the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical assistant that provides accurate, factual information based only on the provided context. Never make up information or use external knowledge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=300,   # Limit response length
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm sorry, I couldn't generate a response based on the provided context."
    
    def generate_fallback(self, query: str) -> str:
        """Generate a fallback response when no relevant contexts are found
        
        Args:
            query: The user's query
            
        Returns:
            Fallback response
        """
        return "I don't have enough information in my knowledge base to answer this question accurately. Please consult a medical professional for specific medical advice."