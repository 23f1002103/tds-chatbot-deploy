import os
from flask import Flask, request, jsonify, render_template # Added render_template
from dotenv import load_dotenv

# LangChain and Google GenAI imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.') # Set template_folder to current directory
load_dotenv() # Load environment variables from .env file

# --- Configuration ---
VECTOR_DB_DIR = "chroma_db"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
# IMPORTANT: Use the model that works for you, likely "gemini-1.5-flash"
GOOGLE_LLM_MODEL = "gemini-1.5-flash" 

# --- Global Components (Initialized once when the API starts) ---
embeddings_model = None
vectorstore = None
retriever = None
llm = None
retrieval_chain = None

def initialize_chatbot_components():
    global embeddings_model, vectorstore, retriever, llm, retrieval_chain

    print("--- Initializing RAG Chatbot Components for API ---")

    # Check API Key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

    # 1. Initialize Google Embedding Model
    print("⏳ Initializing Google embedding model...")
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
    except Exception as e:
        print(f"API Error: Failed to initialize Google Embedding model: {e}")
        raise # Re-raise to prevent app from starting if critical component fails

    print("✅ Google Embedding model loaded.")

    # 2. Load ChromaDB
    print(f"⏳ Loading ChromaDB from {VECTOR_DB_DIR}...")
    try:
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        print(f"API Error: Failed to load ChromaDB: {e}")
        raise # Re-raise to prevent app from starting if critical component fails
    print("✅ ChromaDB loaded and retriever created.")

    # 3. Initialize Google LLM
    print(f"⏳ Initializing Google LLM: {GOOGLE_LLM_MODEL}...")
    try:
        llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.3)
    except Exception as e:
        print(f"API Error: Failed to initialize Google LLM: {e}. Check your GOOGLE_API_KEY and access to '{GOOGLE_LLM_MODEL}'.")
        raise # Re-raise to prevent app from starting if critical component fails
    print("✅ Google LLM loaded.")

    # 4. Define the Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the provided context.
    If you don't know the answer based on the context, politely state that you don't have enough information.
    Keep your answers concise and directly relevant to the context.

    Context:
    {context}

    Question:
    {input}
    """)

    # 5. Create the RAG Chain
    document_combiner = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_combiner)
    print("✅ RAG Chain initialized for API.")

# Initialize components when the Flask app starts
with app.app_context():
    initialize_chatbot_components()

# --- NEW: Route to serve the HTML frontend ---
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
        # Invoke the RAG chain
        response = retrieval_chain.invoke({"input": user_question})
        bot_answer = response["answer"]
        
        return jsonify({
            "question": user_question,
            "answer": bot_answer
        }), 200

    except Exception as e:
        print(f"Error during API chat processing: {e}")
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    print("\n--- Starting Flask API Server ---")
    print("Frontend will be accessible at: http://127.0.0.1:5000/")
    print("API endpoint at: http://127.0.0.1:5000/chat (POST requests)")
    app.run(host='0.0.0.0', port=5000, debug=True)
