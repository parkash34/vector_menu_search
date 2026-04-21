# Bella Italia — Vector Menu Search

A vector database powered menu search system built with FastAPI,
LangChain and Pinecone. Uses semantic search with metadata filtering
to find menu items by meaning and filter by category, diet and price.
Embeddings are stored permanently in Pinecone cloud database.

## Features

- Pinecone vector database — embeddings stored permanently in cloud
- Semantic search — finds meaning not just keywords
- Metadata filtering — filter by category, diet and price
- RAG endpoint — AI answers from retrieved menu context
- HuggingFace embeddings — free local embedding model
- One time embedding — documents embedded once stored forever
- Input validation — empty queries rejected automatically
- Price aware answers — AI sees prices in context for accurate responses

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend web framework |
| LangChain | AI and RAG framework |
| Pinecone | Cloud vector database |
| HuggingFace | Free embedding model |
| Groq API | AI language model |
| LLaMA 3.3 70B | AI model |
| Pydantic | Data validation |
| python-dotenv | Environment variable management |

## Project Structure
```
vector-menu-search/
│
├── env/
├── main.py
├── .env
└── requirements.txt
```

## Setup

1. Clone the repository
```
git clone https://github.com/yourusername/vector-menu-search
```

2. Create and activate virtual environment
```
python -m venv env
env\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
pip install sentence-transformers
```

4. Create `.env` file and add your API keys
```
API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

5. Run the server
```
uvicorn main:app --reload
```

## API Endpoints

### POST /search
```
Basic semantic search — returns top 3 results with similarity scores.
```

**Request:**
```json
{
    "query": "vegan food"
}
```

**Response:**
```json
[
    {
        "text": "Vegan Arrabbiata - spicy tomato pasta with no animal products.",
        "score": 0.51
    }
]
```

### POST /search-filtered
Semantic search with metadata filters.

**Request:**
```json
{
    "query": "vegan food",
    "diet": "vegan",
    "max_price": 13
}
```

**Response:**
```json
[
    {
        "page_content": "Vegan Arrabbiata - spicy tomato pasta.",
        "metadata": {
            "category": "pasta",
            "diet": "vegan",
            "price": 12.0
        }
    }
]
```

### POST /ask
Full RAG endpoint — AI answers using retrieved menu context.

**Request:**
```json
{
    "query": "What is the cheapest dessert?"
}
```

**Response:**
```json
{
    "answer": "The cheapest dessert is Gelato, priced at $5."
}
```

## How It Works
```
Menu documents with metadata
↓
HuggingFace converts to 384 dimension embeddings
↓
Stored permanently in Pinecone cloud index
↓
User sends query
↓
Query converted to embedding
↓
Pinecone finds similar vectors using cosine similarity
↓
Optional metadata filters applied
↓
Relevant chunks sent to AI as context
↓
AI generates accurate answer
```

## Metadata Schema

Each menu item stored with:
```python
{
    "category": "pizza/pasta/dessert/drink",
    "price": 12,
    "diet": "vegan/vegetarian/none",
    "spicy": True/False
}
```

## Filter Options

```python
# Filter by category
{"query": "food", "category": "pasta"}

# Filter by diet
{"query": "food", "diet": "vegan"}

# Filter by price
{"query": "food", "max_price": 10}

# Combined filters
{"query": "food", "diet": "vegan", "max_price": 13}
```

## Similarity Scores
```
Pinecone uses cosine similarity — higher is better:
0.8 - 1.0  →  very relevant
0.5 - 0.8  →  relevant
0.0 - 0.5  →  somewhat related
```
## Environment Variables
```
API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Notes

- Never commit your .env file to GitHub
- Documents embedded once and stored permanently
- Pinecone free tier — 1 index, 2GB storage, no credit card
- Re-embedding only happens if Pinecone index is empty
- Menu updated by modifying menu_documents in main.py