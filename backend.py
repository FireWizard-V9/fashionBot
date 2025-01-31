import openai
from phi.agent import Agent
from phi.llm.openai import OpenAIChat
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.crawl4ai_tools import Crawl4aiTools
import boto3
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
from pinecone import Pinecone as pc, ServerlessSpec  # Updated imports for Pinecone
from PIL import Image
import io
import base64

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

# Initialize Pinecone using the correct method
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = "https://fashion-d2ccnly.svc.aped-4627-b74a.pinecone.io"  # Your custom host URL
pinecone_environment = "us-east-1"  # Your Pinecone environment

# Create Pinecone instance
pc = pc(api_key=pinecone_api_key)

# Access the index using the custom host URL
index_name = "fashion"  # Ensure this matches your Pinecone index name
index = pc.Index(index_name)  # Create an instance of the index

# Predefined lists for clothing classification
CLOTHING_TYPES = {"jacket", "coat", "blouse", "shirt", "blazer", "jeans", "pants", "skirt", "dress"}
STYLES = {"long sleeve", "short sleeve", "sleeveless", "casual", "formal", "sporty"}
COLORS = {"black", "white", "red", "blue", "green", "yellow", "brown", "pink", "purple", "gray"}

# Function to extract features from images
def clothes_features_extract(image_bytes):
    """Extract color, type, and style from clothing image."""
    response = rekognition_client.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=10)
    labels = [label["Name"].lower() for label in response.get('Labels', [])]

    detected_color = next((c for c in labels if c in COLORS), "unknown")
    detected_style = next((s for s in labels if s in STYLES), "unknown")
    detected_type = next((t for t in labels if t in CLOTHING_TYPES), "unknown")

    return {
        "color": detected_color,
        "style": detected_style,
        "type": detected_type
    }

# Function to query Pinecone
def search_pinecone(query, top_k=5):
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed(query)
    search_results = index.query(query_embedding, top_k=top_k)  # Use the index for the search
    return search_results

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        # Convert image to base64
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_image

# Define Agents
web_search_agent = Agent(
    name="web-search-agent",
    role="Searches the web for information",
    model=Groq(id="llama3-70b-8192"),  # Updated to llama3-70b-8192
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

web_scrape_agent = Agent(
    name="web-scrape-agent",
    role="Extracts product details from web pages",
    tools=[Crawl4aiTools()],
    model=Groq(id="llama3-70b-8192"),  # Updated to llama3-70b-8192
    instructions=["Retrieve product name, price, description, availability, website name"],
    show_tool_calls=True,
    markdown=True,
)

rag_agent = Agent(
    name="rag-agent",
    role="Answers user prompt by querying Pinecone",
    model=Groq(id="llama3-70b-8192"),  # Updated to llama3-70b-8192
    instructions=["Retrieve documents from Pinecone based on user input and generate a response"],
    show_tool_calls=True,
    markdown=True,
)

image_analysis_agent = Agent(
    name="image-analysis-agent",
    role="Extracts features from clothing images",
    tools=[clothes_features_extract],  # Use the feature extraction function
    instructions=["Identify colors, clothing types, and styles from images"],
    show_tool_calls=True,
)

cloth_recommendation_agent = Agent(
    name="cloth-recommendation-agent",
    role="Recommends clothes based on user preferences",
    model=Groq(id="llama3-70b-8192"),  # Updated to llama3-70b-8192
    instructions=[
        "Given the color, type, and style of a clothing item, recommend matching outfits.",
        "Use fashion guidelines like color theory and seasonal trends.",
        "Suggest complementary accessories such as shoes, bags, and jewelry."
    ],
    show_tool_calls=True,
)

# Chatbot logic based on input type
def chatbot_response(user_input):
    if user_input.startswith("image:"):  # Check if the input is an image
        # Extract the path of the image file
        image_path = user_input[len("image:"):].strip()

        # Convert the image at the path to base64
        base64_image_data = image_to_base64(image_path)
        
        # Now, decode the base64 data and extract features
        image_data = base64.b64decode(base64_image_data)
        features = clothes_features_extract(image_data)
        
        # Generate clothing recommendations
        prompt = f"""
        I have a {features['color']} {features['style']} {features['type']}.
        What should I wear with it? Suggest clothing and accessories that match.
        """
        recommendation = cloth_recommendation_agent.run(prompt)
        return recommendation

    elif user_input.startswith("pdf:"):  # Check if the input is a PDF query
        search_results = search_pinecone(user_input)
        return search_results

    else:  # If the input is plain text
        # Use multiple agents: web search, web scrape, and RAG
        web_search_response = web_search_agent.run(user_input)
        web_scrape_response = web_scrape_agent.run(user_input)
        rag_response = rag_agent.run(user_input)

        # Combine the results from all agents
        combined_response = f"""
        Web Search Results: {web_search_response}
        Web Scrape Results: {web_scrape_response}
        RAG Response: {rag_response}
        """
        return combined_response

# Test the chatbot with both inputs

# Text query
user_input_text = "Tell me about the latest trends in fashion."
print(chatbot_response(user_input_text))

# Image input 
user_input_image = "image:test_img.png" 
print(chatbot_response(user_input_image))
