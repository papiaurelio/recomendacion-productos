
import requests
import pandas as pd
from pinecone_vec import PineconeClient
from sentence_transformers import SentenceTransformer
import math

from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde .env
load_dotenv()


model = SentenceTransformer('all-MiniLM-L6-v2')

def model_embedder(text):
    embedding = model.encode(text)
    return embedding.tolist()

def batch_upsert(p_client, vectors, batch_size=100):
    """Upsert vectors in batches to avoid request size limits"""
    total_vectors = len(vectors)
    num_batches = math.ceil(total_vectors / batch_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_vectors)
        batch = vectors[start_idx:end_idx]
        
        print(f"Upserting batch {i+1}/{num_batches} ({len(batch)} vectors)")
        p_client.upsert_vectors(batch)

# Load and process products
csv_path = "productsVectors.csv"
print("Loading products from CSV...")
products_df = pd.read_csv(csv_path)
products = products_df.to_dict(orient="records")

#La diferencia de este algoritmo es que se trabaja con los tags que se tienen cargados directamente de la propiedad 'tags'

##Al cargar menos palabras, el proceso es menos tardado, pero se pierde informacion y presicion de las recomendaciones

vectors = []
print("Embedding products...")
for product in products:
    text = product['tags']
    print('Tags: ', text)
    
    if pd.isna(text):
        text = ""
    
    vector = model_embedder(text)

    metadata = {
        'name': "NaN" if pd.isna(product['name']) else product['name'],
        # 'description': "NaN" if pd.isna(product['description']) else product['description'],
        'price': "NaN" if pd.isna(product['price']) else product['price'],
        'imageUrl': "NaN" if pd.isna(product['imageUrl']) else product['imageUrl']
    }   

    vectors.append(
        (
            str(product['id']),
            vector,
            metadata
        )
    )

index_name = os.getenv("INDEX_NAME")
namespace = os.getenv("NAMESPACE")
api_key = os.getenv("PINECONE_API_KEY")

print("Creating Pinecone vectors cloud...")
p_client = PineconeClient(api_key=api_key, index_name=index_name)

batch_upsert(p_client, vectors, batch_size=100)
print("OK")