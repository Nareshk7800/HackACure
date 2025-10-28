# HackACure RAG API

A Retrieval-Augmented Generation (RAG) API for the Hack-A-Cure platform that answers complex medical and scientific questions by retrieving relevant textbook snippets and generating evidence-based responses.

## Features

- Semantic retrieval of relevant textbook snippets
- Evidence-based answer generation
- Fast response time (under 60 seconds)
- RESTful API with JSON input/output

## Project Structure

```
.
├── app.py                 # FastAPI application entry point
├── index_textbooks.py     # Script to index textbooks
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── HackACure-Dataset/    # Directory containing medical textbooks
│   └── Dataset/          # PDF textbooks
└── rag/                  # RAG modules
    ├── __init__.py       # Package initialization
    ├── data_processor.py # Textbook processing and chunking
    ├── vector_store.py   # Vector database for semantic search
    └── generator.py      # Response generation using LLM
```

## Setup

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

4. Add your OpenAI API key to the `.env` file

5. Index the textbooks:
   ```bash
   python index_textbooks.py
   ```

6. Start the API server:
   ```bash
   python app.py
   ```

## API Usage

### Query Endpoint

**Endpoint:** `POST /query`

**Request Format:**
```json
{
  "query": "What are the symptoms of myocardial infarction?",
  "top_k": 5
}
```

**Response Format:**
```json
{
  "answer": "Myocardial infarction (heart attack) symptoms include chest pain or discomfort that may radiate to the arm, shoulder, jaw, or back. Other symptoms include shortness of breath, nausea, lightheadedness, and cold sweats.",
  "contexts": [
    "Symptoms of myocardial infarction include chest pain or discomfort that may radiate to the arm, shoulder, jaw, or back.",
    "Other common symptoms of MI include shortness of breath, nausea, lightheadedness, and cold sweats.",
    "Immediate medical attention should be sought if symptoms of myocardial infarction are suspected."
  ]
}
```

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy"
}
```

## Performance Considerations

- The API is designed to complete the entire retrieval and generation process within 60 seconds
- For optimal performance, keep queries concise and specific
- The `top_k` parameter controls the number of context snippets retrieved (default: 5)

## Evaluation Metrics

The RAG API is evaluated based on:

- Relevancy (30%): How relevant the retrieved contexts are to the query
- Correctness (30%): Factual accuracy of the generated answer
- Context Quality (25%): Quality and informativeness of the retrieved snippets
- Faithfulness (15%): How well the answer is grounded in the retrieved contexts