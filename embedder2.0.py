
import requests
import pandas as pd
from pinecone_vec import PineconeClient
from sentence_transformers import SentenceTransformer
import math
from tqdm import tqdm  # Opcional, para mostrar un progreso visual al recorrer

from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde .env
load_dotenv()

##Un modelo mucho mas capaz que trabaja con mas dimesiones 

model = SentenceTransformer('all-distilroberta-v1')


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

# Cargar productos
csv_path = "productsVectors.csv"
print("Loading products from CSV...")
products_df = pd.read_csv(csv_path)
products = products_df.to_dict(orient="records")


# Preparar vectores
vectors = []
print("Embedding products...")
from tqdm import tqdm

# Inicializa una lista para almacenar los vectores generados
vectors = []

# Itera sobre los productos con tqdm para mostrar el progreso
for product in tqdm(products, desc="Procesando productos", total=len(products)):
    # Concatenar 'name' y 'description' en un solo texto
    text = f"{product.get('name', '')} {product.get('description', '')}"
    # print('Texto procesado:', text)
    
    # Manejar valores NaN
    if pd.isna(text):
        text = ""

    # Generar el vector (embedding) usando el modelo
    vector = model_embedder(text)

    # Crear un diccionario de metadata con validación para NaN
    metadata = {
        'name': "NaN" if pd.isna(product.get('name')) else product['name'],
        'price': "NaN" if pd.isna(product.get('price')) else product['price'],
        'imageUrl': "NaN" if pd.isna(product.get('imageUrl')) else product['imageUrl']
    }

    # Agregar los datos a la lista de vectores
    vectors.append(
        (
            str(product['id']),  # Manejo de ID
            vector,  # Embedding generado
            metadata  # Metadata asociada
        )
    )

# Inicializar variable de pinecone
index_name = os.getenv("INDEX_NAME")
namespace = os.getenv("NAMESPACE")
api_key = os.getenv("PINECONE_API_KEY")

print("Creating Pinecone vectors cloud...")
p_client = PineconeClient(api_key=api_key, index_name=index_name)


batch_upsert(p_client, vectors, batch_size=100)
print("OK")