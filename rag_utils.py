import os
import pinecone
import openai
from dotenv import load_dotenv
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import ServerlessSpec

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone 
pc = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-east-1" 
)

index_name = "fashion"

# Create the index if it doesn't already exist
def create_pinecone_index():
    if index_name not in pc.list_indexes().names():
        spec = ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
        pc.create_index(
            name=index_name,
            dimension=1536,  
            metric="cosine", 
            spec=spec
        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        print(f"Index '{index_name}' already exists!")

# Function to store CSV data into Pinecone
def store_csv_in_pinecone(csv_paths):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    all_documents = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            text_chunk = " ".join(row.astype(str).values)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_text = text_splitter.split_text(text_chunk)

            # Convert each split text to a Document object
            documents = [Document(page_content=text) for text in split_text]
            all_documents.extend(documents)

   
    vector_store = Pinecone.from_documents(all_documents, embeddings, index_name=index_name)
    print(f"✅ Data from CSV stored in Pinecone index: {index_name}")

create_pinecone_index()


csv_paths = ["myntra_products_catalog.csv"]
store_csv_in_pinecone(csv_paths)
