import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from api import CustomGoogleGenerativeAIEmbeddings # We can import your custom class from api.py

# --- Configuration ---
load_dotenv()
VECTOR_DB_DIR = "chroma_db"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
TASK_TYPE_QUERY = "retrieval_query"

def debug_question(question_to_test: str):
    """
    This function initializes the components and tests the retrieval for a single question.
    """
    print("-" * 80)
    print(f"Testing Question: \"{question_to_test}\"")
    print("-" * 80)

    # 1. Initialize the Embedding Model
    print("⏳ Initializing embedding model...")
    try:
        embeddings_model = CustomGoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            task_type=TASK_TYPE_QUERY,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("✅ Embedding model loaded.")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return

    # 2. Load the Chroma Vectorstore
    print(f"⏳ Loading ChromaDB from {VECTOR_DB_DIR}...")
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"❌ Error: Database directory not found at '{VECTOR_DB_DIR}'.")
        print("Please make sure you have run the data scraping and ingestion script.")
        return

    try:
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
        print(f"✅ ChromaDB loaded. It contains {vectorstore._collection.count()} documents.")
    except Exception as e:
        print(f"❌ Failed to load ChromaDB: {e}")
        return

    # 3. Embed the test question
    print("⏳ Embedding the test question...")
    query_embedding = embeddings_model.embed_query(question_to_test)
    print("✅ Question embedded.")

    # 4. Perform the similarity search
    print("⏳ Performing similarity search...")
    retrieved_docs = vectorstore.similarity_search_with_score(
        query=question_to_test,
        k=5 # Let's get the top 5 results
    )

    # 5. Print the results
    print("\n--- RETRIEVAL RESULTS ---")
    if not retrieved_docs:
        print("🚨 No documents found!")
    else:
        for i, (doc, score) in enumerate(retrieved_docs):
            print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
            print(f"  Content: {doc.page_content[:400]}...") # Print the first 400 chars
            print(f"  Metadata: {doc.metadata}")
    print("\n--- END OF RESULTS ---\n")


# --- Main execution ---
if __name__ == '__main__':
    # --- Test Failing Question 1 ---
    q1 = "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?"
    debug_question(q1)

    # --- Test Failing Question 2 ---
    q2 = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"
    debug_question(q2)