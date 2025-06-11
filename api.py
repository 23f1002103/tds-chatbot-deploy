import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# LangChain and Google GenAI imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List # Import List for type hinting in the custom class

# --- Custom Embedding Wrapper (NEW) ---
# This class wraps GoogleGenerativeAIEmbeddings to ensure its output
# is always a standard Python list of floats, which ChromaDB expects.
class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Call the original embed_documents method from the parent class
        raw_embeddings = super().embed_documents(texts)
        
        # Ensure the overall result is a list of lists of floats
        processed_embeddings = []
        for single_embedding_raw in raw_embeddings:
            # Check if it's already a list/tuple of floats, or a Repeated object that needs conversion
            if isinstance(single_embedding_raw, (list, tuple)):
                # If it's a nested list like [[...]] (from previous issues), flatten it once
                if len(single_embedding_raw) == 1 and isinstance(single_embedding_raw[0], (list, tuple)):
                    processed_embeddings.append(list(single_embedding_raw[0]))
                else:
                    # Assume it's already a flat list of floats or needs direct conversion
                    processed_embeddings.append(list(single_embedding_raw))
            elif hasattr(single_embedding_raw, '__iter__'): # Catches 'Repeated' objects specifically
                processed_embeddings.append(list(single_embedding_raw))
            else:
                # Fallback for truly unexpected types; should ideally not be hit
                raise TypeError(f"Expected iterable for single embedding, but received: {type(single_embedding_raw)}")
        
        return processed_embeddings

    def embed_query(self, text: str) -> List[float]:
        # Call the original embed_query method from the parent class
        raw_embedding = super().embed_query(text)
        
        # Ensure the query embedding is a flat list of floats
        if isinstance(raw_embedding, (list, tuple)):
            # If it's a nested list like [[...]] (from previous issues), flatten it once
            if len(raw_embedding) == 1 and isinstance(raw_embedding[0], (list, tuple)):
                return list(raw_embedding[0])
            else:
                # Assume it's already a flat list of floats or needs direct conversion
                return list(raw_embedding)
        elif hasattr(raw_embedding, '__iter__'): # Catches 'Repeated' objects specifically
            return list(raw_embedding)
        else:
            # Fallback for truly unexpected types; should ideally not be hit
            raise TypeError(f"Expected iterable for query embedding, but received: {type(raw_embedding)}")


# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.')
load_dotenv()

# --- Configuration ---
VECTOR_DB_DIR = "chroma_db"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
GOOGLE_LLM_MODEL = "gemini-1.5-flash"

# --- Global Components (Initialized once when the API starts) ---
embeddings_model = None
vectorstore = None
llm = None
prompt = None # Storing prompt globally as it's static

def initialize_chatbot_components():
    global embeddings_model, vectorstore, llm, prompt

    print("--- Initializing RAG Chatbot Components for API ---")

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

    # 1. Initialize Custom Google Embedding Model
    print("⏳ Initializing Custom Google embedding model...")
    try:
        embeddings_model = CustomGoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL) # Use custom wrapper
    except Exception as e:
        print(f"API Error: Failed to initialize Custom Google Embedding model: {e}")
        raise

    print("✅ Custom Google Embedding model loaded.")

    # 2. Load ChromaDB
    print(f"⏳ Loading ChromaDB from {VECTOR_DB_DIR}...")
    try:
        # Ensure embedding_function uses the custom wrapper
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
    except Exception as e:
        print(f"API Error: Failed to load ChromaDB: {e}")
        raise
    print("✅ ChromaDB loaded.")

    # 3. Initialize Google LLM
    print(f"⏳ Initializing Google LLM: {GOOGLE_LLM_MODEL}...")
    try:
        llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.3)
    except Exception as e:
        print(f"API Error: Failed to initialize Google LLM: {e}. Check your GOOGLE_API_KEY and access to '{GOOGLE_LLM_MODEL}'.")
        raise
    print("✅ Google LLM loaded.")

    # 4. Define the Prompt Template (stored globally)
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the provided context.
    If you don't know the answer based on the context, politely state that you don't have enough information.
    Keep your answers concise and directly relevant to the context.

    Context:
    {context}

    Question:
    {input}
    """)
    print("✅ Prompt template defined.")

# Initialize components when the Flask app starts
with app.app_context():
    initialize_chatbot_components()

# --- Route to serve the HTML frontend ---
@app.route('/')
def serve_frontend():
    """Serves the main chatbot HTML page."""
    return render_template('index.html')

# --- API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    API endpoint to receive a user question and return a chatbot answer.
    Expects a JSON payload: {"question": "Your question here?"}
    Returns a JSON payload: {"question": "...", "answer": "..."}
    """
    user_data = request.get_json()

    if not user_data or 'question' not in user_data:
        return jsonify({"error": "Invalid request: Please provide a 'question' in the JSON body."}), 400

    user_question = user_data.get('question')
    print(f"\nReceived API question: '{user_question}'")

    try:
        # Step 1: The embeddings_model (now CustomGoogleGenerativeAIEmbeddings)
        # will ensure the query embedding is in the correct format.
        query_embedding = embeddings_model.embed_query(user_question)
        
        # We no longer need the complex type checking here, as the custom wrapper
        # ensures `query_embedding` is always a flat list of floats.


        # Step 2: Perform similarity search using the flattened embedding
        retrieved_docs = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=4 # Retrieve 4 relevant documents for context
        )
        print(f"Found {len(retrieved_docs)} relevant documents.")

        # Step 3: Prepare the context for the LLM
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        if not context_text:
            context_text = "No specific relevant information found in the knowledge base."
            print("Warning: No context generated from retrieved documents.")

        # Step 4: Format the prompt and invoke the LLM
        formatted_prompt = prompt.format(context=context_text, input=user_question)
        print("Invoking LLM with formatted prompt.")
        
        rag_chain = (
            {"context": lambda x: context_text, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser() # Ensure output is a string
        )
        
        bot_answer = rag_chain.invoke(user_question)
        print("LLM responded successfully.")

        return jsonify({
            "question": user_question,
            "answer": bot_answer
        }), 200

    except Exception as e:
        print(f"Error during API chat processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    print("\n--- Starting Flask API Server ---")
    print("Frontend will be accessible at: http://127.0.0.1:5000/")
    print("API endpoint at: http://127.0.0.1:5000/chat (POST requests)")
    app.run(host='0.0.0.0', port=5000, debug=True)
