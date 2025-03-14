from openai import OpenAI
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
import logging
from dotenv import load_dotenv  
load_dotenv()


# Initialize Pinecone client
uri = os.getenv("MONGO_DB_CONNECTION")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Define index name and dimension
dimension = 1536 


# Function to generate embeddings using text-embedding-3-large
def get_embeddings(texts):
    
    response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
    )
    return response.data[0].embedding

def upsert_mongo(embedvalue, original):
    client = MongoClient(uri, server_api=ServerApi('1'))    
    db = client['finbloom_dev']
    collection = db['account_types_new']
    chat_dict = {
        "type": original.get("type"),
        "rate": original.get("rate"),
        "term": original.get("term"),
        "embedvalue": embedvalue
    }
    result = collection.insert_one(chat_dict)


# Example usage
texts = [{
        "id":1,
        "type":"savings",
          "rate":"savings interest rate is 3.5",
          "term":"6 months"},
          {
              "id":2,
              "type":"CD",
          "rate":"CD interest rate is 4.5",
          "term":"No terms"},
          {
              "id":3,
              "type":"checking",
          "rate":"checking interest rate is 1.2",
          "term":"No terms"}]
for text in texts:
    embeddings = get_embeddings(text.get("rate"))
    
    #upsert_mongo(embeddings, text) 

def vector_search(user_query):
    """
    Perform a vector search in the MongoDB collection based on the user query.
    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.
    Returns:
    list: A list of matching documents.
    """
    client = MongoClient(uri, server_api=ServerApi('1'))    
    db = client['finbloom_dev']
    collection = db['account_types_new']
    # Generate embedding for the user query
    query_embedding = get_embeddings(user_query)
    if query_embedding is None:
        return "Invalid query or embedding generation failed."
    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedvalue",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 3  # Return top 1 matches
            }
        }
    ]
    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

# Querying the index
query_text = "what is savings rate?"

results = vector_search(query_text)

for res in results:
    print(res.get("type")) 
    print(res.get("rate"))
    print(res.get("term"))
    #print(res)
    print("\n\n")

    