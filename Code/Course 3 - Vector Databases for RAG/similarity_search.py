# Importing the necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
import chromadb
from chromadb.utils import embedding_functions

# Define the embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a new instance of ChromaClient to interact with the Chroma DB
client = chromadb.Client()

# Define the name for the collection to be created or retrieved
collection_name = "my_grocery_collection"


# Function to perform a similarity search in the collection
def perform_similarity_search(collection, query_terms, n_results=3):
    try:
        if isinstance(query_terms, str):
            query_terms = [query_terms]

        # Query for the most similar documents to each query term
        results = collection.query(
            query_texts=query_terms,
            n_results=n_results
        )

        # Check if no results are returned or if the results array is empty
        if not results or not results["ids"] or all(len(ids) == 0 for ids in results["ids"]):
            print(f'No documents found similar to "{query_terms}"')
            return

        # 'results' holds one list per query term, so iterate per query
        for q, term in enumerate(query_terms):
            print(f'Top {n_results} similar documents to "{term}":')
            for doc_id, text, score in zip(
                results["ids"][q],
                results["documents"][q],
                results["distances"][q],
            ):
                print(f' - ID: {doc_id}, Text: "{text or "Text not available"}", Score: {score:.4f}')
    except Exception as error:
        print(f"Error in similarity search: {error}")


# Define the main function to interact with the Chroma DB
def main():
    try:
        # Create (or fetch, if it already exists) a collection with a cosine
        # distance metric and the embedding function defined above
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing grocery data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")

        # Array of grocery-related text items
        texts = [
            "fresh red apples",
            "organic bananas",
            "ripe mangoes",
            "whole wheat bread",
            "farm-fresh eggs",
            "natural yogurt",
            "frozen vegetables",
            "grass-fed beef",
            "free-range chicken",
            "fresh salmon fillet",
            "aromatic coffee beans",
            "pure honey",
            "golden apple",
            "red fruit",
        ]

        # Unique IDs in the format 'food_<index>', starting from 1
        ids = [f"food_{index + 1}" for index in range(len(texts))]

        # Insert the documents; ChromaDB generates embeddings with the
        # configured embedding function. upsert keeps reruns idempotent.
        collection.upsert(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )

        # Retrieve all items stored in the collection for inspection
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        perform_similarity_search(collection, ["red", "fresh"], n_results=3)
    except Exception as error:  # Catch any errors and log them to the console
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
