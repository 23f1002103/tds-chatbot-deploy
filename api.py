# import os
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS # Import CORS
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


# # --- Custom Embedding Wrapper ---
# class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
#     def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
#         raw_embeddings = super().embed_documents(texts, **kwargs)
#         processed_embeddings = []
#         for single_embedding_raw in raw_embeddings:
#             if isinstance(single_embedding_raw, (list, tuple)):
#                 if len(single_embedding_raw) == 1 and isinstance(single_embedding_raw[0], (list, tuple)):
#                     processed_embeddings.append(list(single_embedding_raw[0]))
#                 else:
#                     processed_embeddings.append(list(single_embedding_raw))
#             elif hasattr(single_embedding_raw, '__iter__'):
#                 processed_embeddings.append(list(single_embedding_raw))
#             else:
#                 raise TypeError(f"Expected iterable for single embedding, but received: {type(single_embedding_raw)}")
#         return processed_embeddings

#     def embed_query(self, text: str, **kwargs: Any) -> List[float]:
#         raw_embedding = super().embed_query(text, **kwargs)
#         if isinstance(raw_embedding, (list, tuple)):
#             if len(raw_embedding) == 1 and isinstance(raw_embedding[0], (list, tuple)):
#                 return list(raw_embedding[0])
#             else:
#                 return list(raw_embedding)
#         elif hasattr(raw_embedding, '__iter__'):
#             return list(raw_embedding)
#         else:
#             raise TypeError(f"Expected iterable for query embedding, but received: {type(raw_embedding)}")


# # --- Flask App Initialization ---
# app = Flask(__name__, template_folder='.')
# CORS(app) # Enable CORS for all routes by default
# load_dotenv()

# # --- Configuration ---
# VECTOR_DB_DIR = "chroma_db"
# GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
# GOOGLE_LLM_MODEL = "gemini-1.5-flash"

# TASK_TYPE_DOCUMENT = "retrieval_document"
# TASK_TYPE_QUERY = "retrieval_query"

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # --- Global Components (Initialized once when the API starts) ---
# embeddings_model = None
# vectorstore = None
# llm = None
# prompt = None

# def initialize_chatbot_components():
#     global embeddings_model, vectorstore, llm, prompt

#     print("--- Initializing RAG Chatbot Components for API ---")

#     if not os.getenv("GOOGLE_API_KEY"):
#         raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

#     print(f"⏳ Initializing Custom Google embedding model for queries: {GOOGLE_EMBEDDING_MODEL} with task_type='{TASK_TYPE_QUERY}'...")
#     try:
#         embeddings_model = CustomGoogleGenerativeAIEmbeddings(
#             model=GOOGLE_EMBEDDING_MODEL,
#             task_type=TASK_TYPE_QUERY
#         )
#     except Exception as e:
#         print(f"API Error: Failed to initialize Custom Google Embedding model: {e}")
#         raise

#     print("✅ Custom Google Embedding model loaded.")

#     print(f"⏳ Loading ChromaDB from {VECTOR_DB_DIR}...")
#     try:
#         vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
#     except Exception as e:
#         print(f"API Error: Failed to load ChromaDB: {e}")
#         raise
#     print("✅ ChromaDB loaded.")

#     print(f"⏳ Initializing Google LLM: {GOOGLE_LLM_MODEL} for multimodal support...")
#     try:
#         llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.3)
#     except Exception as e:
#         print(f"API Error: Failed to initialize Google LLM: {e}. Check your GOOGLE_API_KEY and access to '{GOOGLE_LLM_MODEL}'.")
#         raise
#     print("✅ Google LLM loaded.")

#     prompt = ChatPromptTemplate.from_template("""
#     Answer the user's question based on the provided context.
#     If you don't know the answer based on the context, politely state that you don't have enough information.
#     Keep your answers concise and directly relevant to the context.

#     Context:
#     {context}

#     Question:
#     {input}
#     """)
#     print("✅ Prompt template defined.")

# with app.app_context():
#     initialize_chatbot_components()

# @app.route('/')
# def serve_frontend():
#     """Serves the main chatbot HTML page."""
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def root_post_endpoint():
#     return chat_endpoint()

# @app.route('/chat', methods=['POST'])
# def chat_endpoint():
#     user_data = request.get_json()

#     if not user_data or 'question' not in user_data:
#         return jsonify({"error": "Invalid request: Please provide a 'question' in the JSON body."}), 400

#     user_question = user_data.get('question')
#     base64_image = user_data.get('image')
#     print(f"\nReceived API question: '{user_question}'")
#     if base64_image:
#         print("Image data received in request.")

#     try:
#         query_embedding = embeddings_model.embed_query(user_question)
        
#         retrieved_docs = vectorstore.similarity_search_by_vector(
#             embedding=query_embedding,
#             k=4
#         )
#         print(f"Found {len(retrieved_docs)} relevant documents.")

#         context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
#         links = []
#         for doc in retrieved_docs:
#             url = doc.metadata.get('url')
#             text = doc.metadata.get('title', doc.metadata.get('topic_title', doc.metadata.get('source', 'Relevant Document')))
#             if url and text:
#                 links.append({"url": url, "text": text})

#         if not context_text:
#             context_text = "No specific relevant information found in the knowledge base."
#             print("Warning: No context generated from retrieved documents.")

#         prompt_parts = []
#         if base64_image:
#             try:
#                 image_data = base64.b64decode(base64_image)
#                 image_stream = BytesIO(image_data)
#                 img = Image.open(image_stream)
                
#                 prompt_parts.append(img)
#                 print("Successfully processed image for multimodal input.")

#             except Exception as img_e:
#                 print(f"Error processing image: {img_e}")
#                 prompt_parts.append(f"Error processing image: {img_e}. Proceeding with text only.")

#         prompt_content = prompt.format(context=context_text, input=user_question)
#         prompt_parts.append(prompt_content)
        
#         print("Invoking Multimodal LLM with prompt parts.")
#         model_multimodal = genai.GenerativeModel(GOOGLE_LLM_MODEL)
        
#         generation_config = {"temperature": 0.3}
#         safety_settings = [
#             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
#         ]

#         try:
#             response = model_multimodal.generate_content(
#                 prompt_parts,
#                 generation_config=generation_config,
#                 safety_settings=safety_settings,
#                 request_options={"timeout": 25} 
#             )
#             bot_answer = response.text
#             print("LLM responded successfully.")
#         except Exception as llm_e:
#             print(f"Error invoking multimodal LLM: {llm_e}")
#             bot_answer = f"I apologize, I encountered an issue processing your request: {llm_e}. Please try again later."
#             links = [] 

#         return jsonify({
#             "answer": bot_answer,
#             "links": links
#         }), 200

#     except Exception as e:
#         print(f"Error during API chat processing: {e}")
#         traceback.print_exc()
#         return jsonify({"error": "An internal server error occurred.", "details": str(e), "links": []}), 500

# if __name__ == '__main__':
#     print("\n--- Starting Flask API Server ---")
#     print("Frontend will be accessible at: http://127.0.0.1:5000/")
#     print("API endpoint at: http://127.0.0.1:5000/chat (POST requests), also root '/' for promptfoo")
#     app.run(host='0.0.0.0', port=5000, debug=True)

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
        print("DEBUG: Entering embed_documents for embedding.") # DEBUG PRINT
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
        print("DEBUG: Finished embed_documents.") # DEBUG PRINT
        return processed_embeddings

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        print(f"DEBUG: Entering embed_query for text: {text[:50]}...") # DEBUG PRINT
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
        print("DEBUG: Finished embed_query.") # DEBUG PRINT


# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.')
CORS(app)
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
        llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, temperature=0.3)
    except Exception as e:
        print(f"API Error: Failed to initialize Google LLM: {e}. Check your GOOGLE_API_KEY and access to '{GOOGLE_LLM_MODEL}'.")
        raise
    print("✅ Google LLM loaded.")

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
    print("DEBUG: Chatbot components initialization complete.") # DEBUG PRINT

with app.app_context():
    initialize_chatbot_components()

@app.route('/')
def serve_frontend():
    print("DEBUG: Serving frontend (GET /)") # DEBUG PRINT
    return render_template('index.html')

@app.route('/', methods=['POST'])
def root_post_endpoint():
    print("DEBUG: POST request received at root (/)") # DEBUG PRINT
    return chat_endpoint()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    print("DEBUG: Entering chat_endpoint (POST /chat)") # DEBUG PRINT
    user_data = request.get_json()

    if not user_data or 'question' not in user_data:
        print("DEBUG: Invalid request: no question found.") # DEBUG PRINT
        return jsonify({"error": "Invalid request: Please provide a 'question' in the JSON body."}), 400

    user_question = user_data.get('question')
    base64_image = user_data.get('image')
    print(f"DEBUG: Received API question: '{user_question}'") # DEBUG PRINT
    if base64_image:
        print("DEBUG: Image data received in request.") # DEBUG PRINT

    try:
        print("DEBUG: Starting query embedding.") # DEBUG PRINT
        query_embedding = embeddings_model.embed_query(user_question)
        print("DEBUG: Query embedding complete. Starting similarity search.") # DEBUG PRINT
        
        retrieved_docs = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=4
        )
        print(f"DEBUG: Found {len(retrieved_docs)} relevant documents.") # DEBUG PRINT

        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        links = []
        for doc in retrieved_docs:
            url = doc.metadata.get('url')
            text = doc.metadata.get('title', doc.metadata.get('topic_title', doc.metadata.get('source', 'Relevant Document')))
            if url and text:
                links.append({"url": url, "text": text})

        if not context_text:
            context_text = "No specific relevant information found in the knowledge base."
            print("DEBUG: Warning: No context generated from retrieved documents.") # DEBUG PRINT

        prompt_parts = []
        if base64_image:
            print("DEBUG: Processing image for multimodal input.") # DEBUG PRINT
            try:
                image_data = base64.b64decode(base64_image)
                image_stream = BytesIO(image_data)
                img = Image.open(image_stream)
                
                prompt_parts.append(img)
                print("DEBUG: Successfully processed image for multimodal input.") # DEBUG PRINT

            except Exception as img_e:
                print(f"DEBUG: Error processing image: {img_e}") # DEBUG PRINT
                prompt_parts.append(f"Error processing image: {img_e}. Proceeding with text only.")

        prompt_content = prompt.format(context=context_text, input=user_question)
        prompt_parts.append(prompt_content)
        
        print("DEBUG: Invoking Multimodal LLM with prompt parts.") # DEBUG PRINT
        model_multimodal = genai.GenerativeModel(GOOGLE_LLM_MODEL)
        
        generation_config = {"temperature": 0.3}
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
            print("DEBUG: LLM responded successfully.") # DEBUG PRINT
        except Exception as llm_e:
            print(f"DEBUG: Error invoking multimodal LLM: {llm_e}") # DEBUG PRINT
            bot_answer = f"I apologize, I encountered an issue processing your request: {llm_e}. Please try again later."
            links = [] 

        print("DEBUG: Preparing JSON response.") # DEBUG PRINT
        return jsonify({
            "answer": bot_answer,
            "links": links
        }), 200

    except Exception as e:
        print(f"DEBUG: Top-level error during API chat processing: {e}") # DEBUG PRINT
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e), "links": []}), 500

if __name__ == '__main__':
    print("\n--- Starting Flask API Server ---")
    print("Frontend will be accessible at: http://127.0.0.1:5000/")
    print("API endpoint at: http://127.0.0.1:5000/chat (POST requests), also root '/' for promptfoo")
    app.run(host='0.0.0.0', port=5000, debug=True)
