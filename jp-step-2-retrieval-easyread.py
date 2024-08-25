import chromadb
from sentence_transformers import SentenceTransformer
import os

# Define the absolute path for persistence directory
persist_dir = "C:/engramar/projects/ai/chromadb/jp-chroma_db_storage"

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Chroma client with local persistence
chroma_client = chromadb.PersistentClient(path=persist_dir)

# Check if the directory exists
if not os.path.exists(persist_dir):
    print(f"Directory does not exist: {persist_dir}")
    exit()

# Get the collection
print("Retrieving the collection...")
collection = chroma_client.get_collection(name="my_test_collection")
if collection is None:
    print("Collection not found.")
    exit()
print("Collection retrieved.")

# Query text
query_text = "What is a Statutory Declaration?"

# Generate embedding for the query
print("Generating query embedding...")
query_embedding = model.encode([query_text])
print("Query embedding...", query_embedding)
print("Query embedding generated.")

# Perform the query
print("Performing the query...")
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3  # Number of results to return
)

# Process and format results
def format_results(results):
    # Extracting lists from the results dictionary
    ids = results.get('ids', [])[0]
    distances = results.get('distances', [])[0]
    documents = results.get('documents', [])[0]

    # Format the results
    formatted = []
    for doc_id, distance, document in zip(ids, distances, documents):
        formatted.append({
            'id': doc_id,
            'distance': distance,
            'document': document
        })
    return formatted

# Format and print the results in sentences
formatted_results = format_results(results)

print("\nHere are the most relevant sentences related to your query:\n")
for result in formatted_results:
    sentences = result['document'].strip().split('. ')
    for sentence in sentences:
        print(f"{sentence.strip()}.")

