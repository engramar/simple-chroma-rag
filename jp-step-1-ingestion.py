import chromadb
from sentence_transformers import SentenceTransformer
import os
import fitz  # PyMuPDF

# Define the absolute path for persistence directory
persist_dir = "C:/engramar/projects/ai/chromadb/jp-chroma_db_storage"
pdf_path = "C:/engramar/projects/ai/chromadb/jp-handbook-full.pdf"

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path=persist_dir)

# Check if the persistence directory exists
if not os.path.exists(persist_dir):
    print(f"Directory does not exist: {persist_dir}")
else:
    print(f"Directory exists: {persist_dir}")

# Extract text from the PDF
print("Extracting text from the PDF...")
with fitz.open(pdf_path) as pdf:
    documents = []
    for page in pdf:
        text = page.get_text()
        documents.append(text)

# Check if any text was extracted
if not documents:
    print("No text was extracted from the PDF.")
else:
    print(f"Extracted {len(documents)} pages of text.")

# Generate embeddings for the extracted text
print("Generating embeddings...")
embeddings = model.encode(documents)
print("Embeddings generated.")

# Get or create the collection
print("Creating or getting the collection...")
collection = chroma_client.get_or_create_collection(name="my_test_collection")
print("Collection created or retrieved.")

# Upsert documents with their embeddings into the collection
print("Upserting documents...")
collection.upsert(
    documents=documents,
    ids=[f"page_{i+1}" for i in range(len(documents))],  # Use page numbers as IDs
    embeddings=embeddings.tolist()    
)
print("Documents upserted.")

# Confirm files in the directory
print("Files in the directory:")
print(os.listdir(persist_dir))
