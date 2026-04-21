import os
import requests
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document


load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API KEY is missing in .env file")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE API KEY is missing in .env file")

app = FastAPI()

class SearchQuery(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_is_empty(cls, v):
        if not v.strip():
            raise ValueError("Query is Empty")
        return v
    
class FilteredQuery(BaseModel):
    query: str
    category: str = None
    max_price: int = None
    diet: str = None

    @field_validator("query")
    @classmethod
    def query_is_empty(cls, v):
        if not v.strip():
            raise ValueError("Query is Empty")
        return v
    

menu_documents = [
    Document(
        page_content="Margherita Pizza - classic tomato sauce and fresh mozzarella.",
        metadata={"category": "pizza", "price": 12, "diet": "vegetarian", "spicy": False}
    ),
    Document(
        page_content="Pepperoni Pizza - spicy pepperoni with mozzarella cheese.",
        metadata={"category": "pizza", "price": 14, "diet": "none", "spicy": True}
    ),
    Document(
        page_content="Vegetarian Pizza - mixed vegetables with tomato sauce.",
        metadata={"category": "pizza", "price": 13, "diet": "vegetarian", "spicy": False}
    ),
    Document(
        page_content="Spicy Chicken Pizza - grilled chicken with hot sauce.",
        metadata={"category": "pizza", "price": 15, "diet": "none", "spicy": True}
    ),
    Document(
        page_content="Carbonara - creamy pasta with bacon and egg.",
        metadata={"category": "pasta", "price": 13, "diet": "none", "spicy": False}
    ),
    Document(
        page_content="Bolognese - rich meat sauce with spaghetti.",
        metadata={"category": "pasta", "price": 14, "diet": "none", "spicy": False}
    ),
    Document(
        page_content="Vegan Arrabbiata - spicy tomato pasta with no animal products.",
        metadata={"category": "pasta", "price": 12, "diet": "vegan", "spicy": True}
    ),
    Document(
        page_content="Tiramisu - classic Italian coffee dessert.",
        metadata={"category": "dessert", "price": 7, "diet": "vegetarian", "spicy": False}
    ),
    Document(
        page_content="Gelato - homemade Italian ice cream.",
        metadata={"category": "dessert", "price": 5, "diet": "vegetarian", "spicy": False}
    ),
    Document(
        page_content="Water - still or sparkling.",
        metadata={"category": "drink", "price": 2, "diet": "vegan", "spicy": False}
    ),
    Document(
        page_content="Fresh Juice - orange or apple.",
        metadata={"category": "drink", "price": 4, "diet": "vegan", "spicy": False}
    ),
    Document(
        page_content="Wine - red or white.",
        metadata={"category": "drink", "price": 8, "diet": "vegan", "spicy": False}
    )
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=pinecone_api_key)

if "restaurant-menu" not in pc.list_indexes().names():
    pc.create_index(
        name="restaurant-menu",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

vector_store = PineconeVectorStore(
    index_name="restaurant-menu",
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index("restaurant-menu")
stats = index.describe_index_stats()

if stats.total_vector_count == 0:
    vector_store.add_documents(menu_documents)


llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0.2,
    max_tokens = 500,
    api_key = api_key
)

@app.post("/search")
def searching(query: SearchQuery):
    query = query.query
    results = vector_store.similarity_search_with_score(query, k=3)
    formatted = []

    for doc, score in results:
        if score > 0.0:
            formatted.append({
                "text": doc.page_content,
                "score": float(score)
            })
    return formatted

@app.post("/search-filtered")
def filtered_searching(filteredQuery: FilteredQuery):
    filter_dic = {}

    if filteredQuery.category:
        filter_dic["category"] = filteredQuery.category
    if filteredQuery.diet:
        filter_dic["diet"] = filteredQuery.diet
    if filteredQuery.max_price:
        filter_dic["price"] = {"$lte" : int(filteredQuery.max_price)}


    results = vector_store.similarity_search(
        query=filteredQuery.query,
        k=3,
        filter=filter_dic
    )
    return results


@app.post("/ask")
def menu_ai(query: SearchQuery):
    query = query.query

    results = vector_store.similarity_search(query, k=3)
    context = ""
    for doc in results:
        price = doc.metadata.get("price", "")
        context += f"{doc.page_content} Price: ${price}\n"

    prompt = f"""You are a restaurant assistant for Bella Italia.
    Answer the customer question using ONLY the menu information below.
    If the answer is not in the menu information say:
    "I don't have that information."

    Menu Information:
    {context}

    Customer Question: {query}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}
