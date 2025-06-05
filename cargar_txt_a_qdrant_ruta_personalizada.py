from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Leer el archivo de texto desde la ruta personalizada
with open(r"D:\Descargas\Codigo 2\Prueba-datos\inteligencia_artificial.txt", "r", encoding="utf-8") as f:
    contenido = f.read()

# Separar el texto por líneas y limpiar
fragmentos = [line.strip() for line in contenido.split("\n") if line.strip()]

# Cargar el modelo de embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")
vectores = modelo.encode(fragmentos)

# Conectar a Qdrant local
client = QdrantClient(host="localhost", port=6333)

# Crear (o recrear) colección
client.recreate_collection(
    collection_name="ia_txt",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Preparar los datos a insertar
puntos = [
    PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload={"texto": t})
    for v, t in zip(vectores, fragmentos)
]

# Insertar en Qdrant
client.upsert(collection_name="ia_txt", points=puntos)

# Crear snapshot
snapshot = client.create_snapshot(collection_name="ia_txt")
print("Snapshot creado:", snapshot.name)
