from pinecone import Pinecone

from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde .env
load_dotenv()


# Inicializar Pinecone
index_name = os.getenv("INDEX_NAME")
namespace = os.getenv("NAMESPACE")
api_key = os.getenv("PINECONE_API_KEY")

print("Conectando a Pinecone...")
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# Borrar todos los vectores
print("Borrando todos los vectores...")
try:
    # Intentar borrar con namespace específico
    # index.delete(delete_all=True, namespace=namespace)
    index.delete(delete_all=True)
except Exception as e:
    print(f"No se pudo borrar el namespace específico: {e}")
    print("Intentando borrar todos los vectores sin especificar namespace...")
    # Borrar todos los vectores sin especificar namespace
    index.delete(delete_all=True)

print("Operación de borrado completada.")

