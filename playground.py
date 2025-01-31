import openai
from phi.agent import Agent
from phi.llm.openai import OpenAIChat
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
import boto3
import base64
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.crawl4ai_tools import Crawl4aiTools
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone as pc

# Load environment variables
load_dotenv()

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# AWS Rekognition Client
rekognition_client = boto3.client(
    "rekognition",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
)

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "fashion"
pc = pc(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Clothing attributes
CLOTHING_TYPES = {"jacket", "coat", "blouse", "shirt", "blazer", "jeans", "pants", "skirt", "dress"}
STYLES = {"long sleeve", "short sleeve", "sleeveless", "casual", "formal", "sporty"}
COLORS = {"black", "white", "red", "blue", "green", "yellow", "brown", "pink", "purple", "gray"}

# 🛠️ Tool: Extract Clothing Features from Image
@tool
def clothes_features_extract(image_base64: str):
    """Extracts color, type, and style from a base64-encoded image."""
    image_bytes = base64.b64decode(image_base64)
    response = rekognition_client.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=10)
    labels = [label["Name"].lower() for label in response.get('Labels', [])]

    return {
        "color": next((c for c in labels if c in COLORS), "unknown"),
        "style": next((s for s in labels if s in STYLES), "unknown"),
        "type": next((t for t in labels if t in CLOTHING_TYPES), "unknown"),
    }

# 🛠️ Tool: Search Pinecone for Related Fashion Documents
@tool
def search_pinecone(query: str, top_k: int = 5):
    """Searches Pinecone for relevant documents based on the query."""
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed(query)
    search_results = index.query(query_embedding, top_k=top_k)
    return search_results

# Agents
web_search_agent = Agent(
    name="web-search-agent",
    role="Searches the web for fashion-related info.",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
)

web_scrape_agent = Agent(
    name="web-scrape-agent",
    role="Extracts product details from web pages.",
    tools=[Crawl4aiTools()],
    model=Groq(id="llama3-70b-8192"),
)

rag_agent = Agent(
    name="rag-agent",
    role="Answers queries using Pinecone vector search.",
    model=Groq(id="llama3-70b-8192"),
    tools=[search_pinecone],
)

cloth_recommendation_agent = Agent(
    name="cloth-recommendation-agent",
    role="Recommends fashion outfits.",
    model=Groq(id="llama3-70b-8192"),
    instructions=["Recommend clothing based on color, type, and style."],
)

# Main Phidata Agent
fashion_ai = Agent(
    name="fashion-ai",
    role="Helps users with fashion advice, image analysis, and trend research.",
    model=Groq(id="llama3-70b-8192"),
    tools=[clothes_features_extract, search_pinecone, DuckDuckGo(), Crawl4aiTools()],
    instructions=[
        "If given a base64 image, extract fashion features.",
        "If given a question, search Pinecone or web.",
    ],
    show_tool_calls=True,
)


fashion_ai.cli()
