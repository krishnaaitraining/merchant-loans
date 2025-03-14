from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
import logging
from dotenv import load_dotenv  
load_dotenv()


# Initialize Pinecone client

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Define index name and dimension
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_KEY")) 
index_name = "finbloom-test" 
dimension = 1536 

# Create Pinecone index if it doesn't exist
if not pinecone_client.has_index(index_name):
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pinecone_client.create_index(index_name, dimension=dimension, metric="cosine", spec=spec) 

# Function to generate embeddings using text-embedding-3-large
def get_embeddings(texts):
    
    response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Upsert data into Pinecone
index = pinecone_client.Index(index_name) 

'''
# Example usage
texts = [{
        "id":1,
        "type":"savings",
          "rate":"savings interest rate is 3.5",
          "term":"6 months"},
          {
        "id":4,
        "type":"savings Promo",
          "rate":"promo savings interest rate is 3.5",
          "term":"6 months"},
          {
              "id":2,
              "type":"CD",
          "rate":"CD interest rate is 4.5",
          "term":"No terms"},
          {
              "id":3,
              "type":"checking",
          "rate":"checking interest rate is 1.2", # rate like '%query%'
          "term":"No terms"}]
for text in texts:
    embeddings = get_embeddings(text.get("rate"))
    index.upsert([{"id": text.get("type"), "values": embeddings, "metadata": text}]) 
'''

# Querying the index
query_text = "which restaurant are you?"
query_embedding = get_embeddings([query_text])
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

print(results) 