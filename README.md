# Recomendación de productos

Este proyecto implementa técnicas avanzadas de procesamiento del lenguaje natural para generar recomendaciones basadas en similitudes entre productos.

## Tecnologías Utilizadas

- **Librería**: `SentenceTransformer`
- **Modelos**:
  - `all-distilroberta-v1`
  - `all-MiniLM-L6-v2`

Estos modelos fueron seleccionados para comparar su rendimiento y precisión al procesar los datos.

## Descripción del Proceso

1. **Pruebas con el Dataset**  
   Se realizaron múltiples pruebas utilizando diferentes subconjuntos de nuestro dataset para evaluar la eficacia de los modelos.

2. **Generación de Embeddings**

   - Se utilizó el algoritmo de **similaridad del coseno** para calcular la similitud entre los vectores generados por los embeddings.
   - Los vectores generados representan las características de los productos en un espacio de alta dimensión.

3. **Almacenamiento de Datos**
   - Los embeddings fueron alojados en **Pinecone**, una plataforma de vector search, para garantizar la disponibilidad y escalabilidad de los datos.

## ¿Por Qué Elegimos Estas Herramientas?

- **SentenceTransformer**: Permite una fácil integración y es ampliamente reconocido por su rendimiento en tareas de NLP.
- **Pinecone**: Ofrece una solución robusta y escalable para gestionar vectores en tiempo real.

---

## Configuración

1. Copia el archivo `.env.example` y renómbralo a `.env`:
   ```bash
   cp .env.example .env
   ```

## Resultado de ejemplo
![image](https://github.com/user-attachments/assets/d9828eb7-1205-44be-b2e0-ed910a591116)


