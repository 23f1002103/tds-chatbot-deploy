import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Import CORS
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import traceback

# LangChain and Google GenAI imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Any


# --- Custom Embedding Wrapper ---
class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        raw_embeddings = super().embed_documents(texts, **kwargs)
        processed_embeddings = []
        for single_embedding_raw in raw_embeddings:
            if isinstance(single_embedding_raw, (list, tuple)):
                if len(single_embedding_raw) == 1 and isinstance(single_embedding_raw[0], (list, tuple)):
                    processed_embeddings.append(list(single_embedding_raw[0]))
                else:
                    processed_embeddings.append(list(single_embedding_raw))
            elif hasattr(single_embedding_raw, '__iter__'):
                processed_embeddings.append(list(single_embedding_raw))
            else:
                raise TypeError(f"Expected iterable for single embedding, but received: {type(single_embedding_raw)}")
        return processed_embeddings

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        raw_embedding = super().embed_query(text, **kwargs)
        if isinstance(raw_embedding, (list, tuple)):
            if len(raw_embedding) == 1 and isinstance(raw_embedding[0], (list, tuple)):
                return list(raw_embedding[0])
            else:
                return list(raw_embedding)
        elif hasattr(raw_embedding, '__iter__'):
            return list(raw_embedding)
        else:
            raise TypeError(f"Expected iterable for query embedding, but received: {type(raw_embedding)}")


# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.')
CORS(app) # Enable CORS for all routes by default
load_dotenv()

# --- Configuration ---
VECTOR_DB_DIR = "chroma_db"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
GOOGLE_LLM_MODEL = "gemini-1.5-flash"

TASK_TYPE_DOCUMENT = "retrieval_document"
TASK_TYPE_QUERY = "retrieval_query"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Global Components (Initialized once when the API starts) ---
embeddings_model = None
vectorstore = None
llm = None
prompt = None

def initialize_chatbot_components():
    global embeddings_model, vectorstore, llm, prompt

    print("--- Initializing RAG Chatbot Components for API ---")

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

    print(f"⏳ Initializing Custom Google embedding model for queries: {GOOGLE_EMBEDDING_MODEL} with task_type='{TASK_TYPE_QUERY}'...")
    try:
        embeddings_model = CustomGoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            task_type=TASK_TYPE_QUERY
        )
    except Exception as e:
        print(f"API Error: Failed to initialize Custom Google Embedding model: {e}")
        raise

    print("✅ Custom Google Embedding model loaded.")

    print(f"⏳ Loading ChromaDB from {VECTOR_DB_DIR}...")
    try:
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
    except Exception as e:
        print(f"API Error: Failed to load ChromaDB: {e}")
        raise
    print("✅ ChromaDB loaded.")

    print(f"⏳ Initializing Google LLM: {GOOGLE_LLM_MODEL} for multimodal support...")
    try:
        llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.1) # Temperature lowered
    except Exception as e:
        print(f"API Error: Failed to initialize Google LLM: {e}. Check your GOOGLE_API_KEY and access to '{GOOGLE_LLM_MODEL}'.")
        raise
    print("✅ Google LLM loaded.")

    # UPDATED: More forceful and direct prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Teaching Assistant for the IIT Madras an AI course. Your personality is helpful, direct, and precise.
    Your primary goal is to answer the user's question using ONLY the 'Context' provided.
    Synthesize an answer directly from the information in the context.
    When you use information from a document in the context, you MUST cite the source by including its 'url' in the 'links' section of your response.
    If the context does not contain the information needed to answer, and only in that case, say "I do not have enough information to answer this question." and provide no links.

    Context:
    {context}

    Question:
    {input}
    """)
    print("✅ Prompt template defined.")

with app.app_context():
    initialize_chatbot_components()

@app.route('/')
def serve_frontend():
    """Serves the main chatbot HTML page."""
    return render_template('index.html')

@app.route('/', methods=['POST'])
def root_post_endpoint():
    return chat_endpoint()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_data = request.get_json()

    if not user_data or 'question' not in user_data:
        return jsonify({"error": "Invalid request: Please provide a 'question' in the JSON body."}), 400

    user_question = user_data.get('question')
    base64_image = user_data.get('image')
    print(f"\nReceived API question: '{user_question}'")
    if base64_image:
        print("Image data received in request.")

    # --- START: NEW QUERY TRANSFORMATION STEP ---
    print("⏳ Transforming user query for better retrieval...")
    query_transform_prompt = f"""
Given the following user question, rewrite it as a concise, keyword-focused query suitable for a vector database search.
Do not answer the question, just reformulate it.

User Question: "{user_question}"

Optimal Query:"""

    try:
        # Use the main LLM to perform the transformation
        transformed_query_response = llm.invoke(query_transform_prompt)
        search_query = transformed_query_response.content
        print(f"✅ Transformed query: '{search_query}'")
    except Exception as e:
        print(f"⚠️ Query transformation failed, using original question. Error: {e}")
        search_query = user_question
    # --- END: NEW QUERY TRANSFORMATION STEP ---

    try:
        # Use the new 'search_query' for embedding, not the original 'user_question'
        query_embedding = embeddings_model.embed_query(search_query)
        
        retrieved_docs = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=4
        )
        print(f"Found {len(retrieved_docs)} relevant documents.")

        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        links = []
        for doc in retrieved_docs:
            url = doc.metadata.get('url')
            text = doc.metadata.get('title', doc.metadata.get('topic_title', doc.metadata.get('source', 'Relevant Document')))
            if url and text:
                links.append({"url": url, "text": text})

        if not context_text:
            context_text = "No specific relevant information found in the knowledge base."
            print("Warning: No context generated from retrieved documents.")

        prompt_parts = []
        if base64_image:
            try:
                image_data = base64.b64decode(base64_image)
                image_stream = BytesIO(image_data)
                img = Image.open(image_stream)
                
                prompt_parts.append(img)
                print("Successfully processed image for multimodal input.")

            except Exception as img_e:
                print(f"Error processing image: {img_e}")
                prompt_parts.append(f"Error processing image: {img_e}. Proceeding with text only.")

        prompt_content = prompt.format(context=context_text, input=user_question)
        prompt_parts.append(prompt_content)
        
        print("Invoking Multimodal LLM with prompt parts.")
        model_multimodal = genai.GenerativeModel(GOOGLE_LLM_MODEL)
        
        # UPDATED: Temperature is now set during llm initialization, but we can redefine here if needed
        generation_config = {"temperature": 0.1}
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        try:
            response = model_multimodal.generate_content(
                prompt_parts,
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={"timeout": 25} 
            )
            bot_answer = response.text
            print("LLM responded successfully.")
        except Exception as llm_e:
            print(f"Error invoking multimodal LLM: {llm_e}")
            bot_answer = f"I apologize, I encountered an issue processing your request: {llm_e}. Please try again later."
            links = [] 

        return jsonify({
            "answer": bot_answer,
            "links": links
        }), 200

    except Exception as e:
        print(f"Error during API chat processing: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e), "links": []}), 500

if __name__ == '__main__':
    print("\n--- Starting Flask API Server ---")
    print("Frontend will be accessible at: http://127.0.0.1:5000/")
    print("API endpoint at: http://127.0.0.1:5000/chat (POST requests), also root '/' for promptfoo")
    app.run(host='0.0.0.0', port=5000, debug=True)

# import os
# from flask import Flask, request, jsonify, render_template, make_response
# from flask_cors import CORS
# from dotenv import load_dotenv
# import base64
# from io import BytesIO
# from PIL import Image
# import google.generativeai as genai
# import traceback

# # LangChain and Google GenAI imports
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from typing import List, Any

# # --- Debug API Key ---
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY environment variable is not set! Please check your .env file.")

# genai.configure(api_key=GOOGLE_API_KEY)

# # --- Flask App Initialization ---
# app = Flask(__name__, template_folder='.')
# CORS(app)

# # --- Configuration ---
# VECTOR_DB_DIR = "chroma_db"
# GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
# GOOGLE_LLM_MODEL = "gemini-1.5-flash"

# TASK_TYPE_DOCUMENT = "retrieval_document"
# TASK_TYPE_QUERY = "retrieval_query"

# # --- Global Components (Initialized once when the API starts) ---
# embeddings_model = None
# vectorstore = None
# llm = None
# prompt = None

# def initialize_chatbot_components():
#     global embeddings_model, vectorstore, llm, prompt
#     print("--- Initializing RAG Chatbot Components ---")

#     print(f"⏳ Initializing Google Embedding Model: {GOOGLE_EMBEDDING_MODEL}...")
#     embeddings_model = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL, task_type=TASK_TYPE_QUERY)
#     print("✅ Google Embedding Model Loaded.")

#     print(f"⏳ Loading ChromaDB from {VECTOR_DB_DIR}...")
#     vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
#     print("✅ ChromaDB Loaded.")

#     print(f"⏳ Initializing Google LLM Model: {GOOGLE_LLM_MODEL}...")
#     llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.3)
#     print("✅ Google LLM Loaded.")

#     prompt = ChatPromptTemplate.from_template("""
#     Answer the user's question based on the provided context.
#     If you don't know the answer based on the context, politely state that you don't have enough information.
#     Keep your answers concise and directly relevant to the context.

#     Context:
#     {context}

#     Question:
#     {input}
#     """)
#     print("✅ Prompt Template Defined.")
#     print("DEBUG: Chatbot initialization complete.")

# with app.app_context():
#     initialize_chatbot_components()

# # --- API Endpoints ---
# @app.route('/chat', methods=['POST'])
# def chat_endpoint():
#     print("DEBUG: POST request received at /chat")
#     user_data = request.get_json()

#     if not user_data or 'question' not in user_data:
#         return jsonify({"error": "Invalid request: Missing 'question' parameter."}), 400

#     user_question = user_data.get('question')
#     base64_image = user_data.get('image')

#     # Validate API Key in Request
#     user_api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
#     if user_api_key != GOOGLE_API_KEY:
#         return jsonify({"error": "Invalid API Key"}), 401

#     print(f"DEBUG: Processing Question: '{user_question}'")
#     if base64_image:
#         print("DEBUG: Image data received in request.")

#     try:
#         print("DEBUG: Generating query embedding...")
#         query_embedding = embeddings_model.embed_query(user_question)

#         print("DEBUG: Performing similarity search...")
#         retrieved_docs = vectorstore.similarity_search_by_vector(embedding=query_embedding, k=4)

#         context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
#         links = [{"url": doc.metadata.get('url', ""), "text": doc.metadata.get('title', "Relevant Document")} for doc in retrieved_docs]

#         if not context_text:
#             context_text = "No specific relevant information found in the knowledge base."

#         prompt_parts = [prompt.format(context=context_text, input=user_question)]
#         if base64_image:
#             try:
#                 image_data = base64.b64decode(base64_image)
#                 image_stream = BytesIO(image_data)
#                 img = Image.open(image_stream)
#                 prompt_parts.append(img)
#                 print("DEBUG: Image successfully processed.")
#             except Exception as img_error:
#                 print(f"DEBUG: Error processing image: {img_error}")

#         print("DEBUG: Invoking Google LLM...")
#         model_multimodal = genai.GenerativeModel(GOOGLE_LLM_MODEL)
#         response = model_multimodal.generate_content(prompt_parts, generation_config={"temperature": 0.3})

#         bot_answer = response.text if response else "Sorry, I couldn't process the request."
#         print("DEBUG: LLM Response Complete.")

#         return jsonify({"answer": bot_answer, "links": links})

#     except Exception as e:
#         print(f"DEBUG: API Error: {e}")
#         return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

# if __name__ == '__main__':
#     print("\n--- Starting Flask API Server ---")
#     print("Frontend accessible at: http://127.0.0.1:5000/")
#     print("API Endpoint at: http://127.0.0.1:5000/chat (POST requests)")
#     app.run(host='0.0.0.0', port=5000, debug=True)
