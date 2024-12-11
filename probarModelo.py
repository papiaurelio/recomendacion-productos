from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde .env
load_dotenv()

class PineconeClient:
    def __init__(self, index_name: str, namespace: str):
        self.index_name = index_name
        self.namespace = namespace
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)

    def query_similarity(self, product_id: str, top_k=10):
        try:
            vector_result = self.index.fetch(ids=[product_id])
            if not vector_result.vectors:
                raise ValueError(f"No se encontró vector para el ID: {product_id}")
            
            product_vector = vector_result.vectors[product_id].values
            
            return self.index.query(
                vector=product_vector,
                top_k=top_k + 1,
                include_values=False,
                include_metadata=True
            )
        except Exception as e:
            raise Exception(f"Error al buscar productos similares: {str(e)}")


def main():
    # Configuración
    INDEX_NAME = os.getenv("INDEX_NAME")
    NAMESPACE = os.getenv("NAMESPACE")
    PRODUCT_ID = "B0032ISERS"  ## Especificio el ID del producto con el cual quiero probar el algoritmo
    TOP_K = 10

    try:
        # Inicializar cliente
        client = PineconeClient(INDEX_NAME, NAMESPACE)
        
        # Realizar búsqueda
        results = client.query_similarity(PRODUCT_ID, TOP_K)
        
        # Mostrar resultados
        print("\nProductos similares encontrados:")
        for match in results.matches:
            print("\n-----------------------------------")
            print(f"ID: {match.id}")
            print(f"Score de similitud: {match.score:.4f}")
            print("Metadata:")
            for key, value in match.metadata.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()