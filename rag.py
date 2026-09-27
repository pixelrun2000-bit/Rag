import requests
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


COLLECTION_NAME = "rag_demo_collection"
DOCUMENT_DIR = "documents"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"


def load_documents():
    documents = []

    for file_path in Path(DOCUMENT_DIR).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        documents.append({
            "filename": file_path.name,
            "text": text
        })

    return documents


def split_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start = end - overlap

        if start >= len(words):
            break

    return chunks


def create_qdrant_collection(qdrant, vector_size):
    collections = qdrant.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if COLLECTION_NAME in collection_names:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )


def store_chunks(qdrant, chunks, embeddings):
    points = []

    for index, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=index,
                vector=embeddings[index],
                payload={
                    "chunk_id": index,
                    "text": chunk
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


def retrieve_context(qdrant, model, question, top_k=3):
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding,
        limit=top_k
    ).points

    retrieved_chunks = []

    for result in results:
        retrieved_chunks.append(result.payload["text"])

    return retrieved_chunks


def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

If the answer is not found in the context, say:
I could not find the answer in the provided document.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt.strip()


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"]


def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(
        host="localhost",
        port=6333
    )

    print("Loading documents...")
    documents = load_documents()

    all_chunks = []

    for document in documents:
        chunks = split_text(document["text"])

        for chunk in chunks:
            all_chunks.append(chunk)

    if len(all_chunks) == 0:
        print("No .txt files found inside the documents folder.")
        return

    print(f"Number of chunks: {len(all_chunks)}")

    print("Creating embeddings...")
    embeddings = embedding_model.encode(
        all_chunks,
        normalize_embeddings=True
    ).tolist()

    vector_size = len(embeddings[0])
    print(f"Vector size: {vector_size}")

    print("Creating Qdrant collection...")
    create_qdrant_collection(qdrant, vector_size)

    print("Storing chunks in Qdrant...")
    store_chunks(qdrant, all_chunks, embeddings)

    print("\nRAG system is ready.")
    print("Ask a question about the documents.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Enter your question: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        retrieved_chunks = retrieve_context(
            qdrant,
            embedding_model,
            question,
            top_k=3
        )

        prompt = build_prompt(question, retrieved_chunks)

        answer = ask_ollama(prompt)

        print("\nAnswer:")
        print(answer)
        print("-" * 40)


main()