import chromadb
from sentence_transformers import SentenceTransformer
import os

# Define the absolute path for persistence directory
persist_dir = "C:/engramar/projects/ai/chromadb/chroma_db_storage"

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path=persist_dir)

# Check if the directory exists
if not os.path.exists(persist_dir):
    print(f"Directory does not exist: {persist_dir}")
else:
    print(f"Directory exists: {persist_dir}")

# Get or create the collection
print("Creating or getting the collection...")
collection = chroma_client.get_or_create_collection(name="my_test_collection")
print("Collection created or retrieved.")

# Documents to upsert
documents = [
    "Test document about apples",
    "Test document about bananas"
]

# Generate embeddings for the documents
print("Generating embeddings...")
embeddings = model.encode(documents)
print("Embeddings generated-------------------->, ", embeddings)

# Upsert documents with their embeddings into the collection
print("Upserting documents...")
collection.upsert(
    documents=documents,
    ids=["test_id1", "test_id2"],
    embeddings=embeddings.tolist()    
)
print("Collection embeddings------------------->, ", embeddings)
print("Documents upserted.")

# Confirm files in the directory
print("Files in the directory:")
print(os.listdir(persist_dir))
