
from pinecone import Pinecone

class PineconeClient:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, vectors: list):
        return self.index.upsert(
            vectors=vectors
        )
    

    #Declaramos esta funcion para traer a los 10 productos mas cercanos

    def query_similarity(self, product_id: str, top_k=10):
    # Primero obtenemos el vector del ID proporcionado
        try:
            vector_result = self.index.fetch(ids=[product_id])
            if not vector_result.vectors:
                raise ValueError(f"No se encontró vector para el ID: {product_id}")
                
            # Extraemos el vector del producto
            product_vector = vector_result.vectors[product_id].values
            
            # Ahora buscamos los vectores similares
            return self.index.query(
                vector=product_vector,
                top_k=top_k + 1,  # Pedimos uno más porque el producto mismo estará en los resultados
                include_values=False,
                include_metadata=True
            )
        except Exception as e:
            raise Exception(f"Error al buscar productos similares: {str(e)}")



    # def query_similarity(self, query_vector: list, top_k=10):
    #     return self.index.query(
    #         vector=query_vector,
    #         top_k=top_k,
    #         include_values=False,
    #         include_metadata=True
    #     )