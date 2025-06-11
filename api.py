import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# LangChain and Google GenAI imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

    # 1. Initialize Google Embedding Model
    print("⏳ Initializing Google embedding model... This might take a moment.")
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
    except Exception as e:
        print(f"API Error: Failed to initialize Google Embedding model: {e}")
        raise

    print("✅ Google Embedding model loaded.")

    # 2. Load ChromaDB
    print(f"⏳ Loading ChromaDB from {VECTOR_DB_DIR}... This also might take a moment.")
    try:
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
        # Step 1: Explicitly embed the user's question
        embedding_result = embeddings_model.embed_query(user_question)

        # --- REVISED EMBEDDING PROCESSING LOGIC ---
        # Ensure the embedding is a standard Python list and is flat.
        # GoogleGenerativeAIEmbeddings.embed_query can sometimes return a
        # 'proto.marshal.collections.repeated.Repeated' object or a nested list.

        query_embedding = []
        if isinstance(embedding_result, (list, tuple)):
            # If it's already a list or tuple:
            if len(embedding_result) == 1 and isinstance(embedding_result[0], (list, tuple)):
                # If it's a list containing a single list (e.g., [[...]]), flatten it
                query_embedding = list(embedding_result[0])
            else:
                # Otherwise, assume it's already a flat list or needs conversion
                query_embedding = list(embedding_result)
        elif hasattr(embedding_result, '__iter__'): # Catches 'Repeated' and other iterable non-list types
            query_embedding = list(embedding_result)
        else:
            raise ValueError(f"Unexpected embedding format: {type(embedding_result)}, value: {embedding_result}")

        # Final check to ensure all elements are numbers
        if not all(isinstance(x, (float, int)) for x in query_embedding):
             raise ValueError(f"Embedding contains non-numeric elements: {query_embedding}")
        # --- END REVISED EMBEDDING PROCESSING LOGIC ---


        # Step 3: Perform similarity search using the flattened embedding
        # We use similarity_search_by_vector which takes the pre-computed embedding
        retrieved_docs = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=4 # Retrieve 4 relevant documents for context
        )
        print(f"Found {len(retrieved_docs)} relevant documents.")

        # Step 4: Prepare the context for the LLM
        # Concatenate the page_content of the retrieved documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        if not context_text:
            context_text = "No specific relevant information found in the knowledge base."
            print("Warning: No context generated from retrieved documents.")

        # Step 5: Format the prompt and invoke the LLM
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
        # Log the full traceback if possible for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    print("\n--- Starting Flask API Server ---")
    print("Frontend will be accessible at: http://127.0.0.1:5000/")
    print("API endpoint at: http://127.0.0.1:5000/chat (POST requests)")
    app.run(host='0.0.0.0', port=5000, debug=True)

