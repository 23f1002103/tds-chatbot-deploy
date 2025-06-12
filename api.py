import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image # Pillow library for image handling
import google.generativeai as genai # For direct Gemini API calls if needed for multimodal
import traceback # Import traceback for detailed error logging

# LangChain and Google GenAI imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Any # Import List and Any for type hinting in the custom class


# --- Custom Embedding Wrapper ---
# This class wraps GoogleGenerativeAIEmbeddings to ensure its output
# is always a standard Python list of floats, which ChromaDB expects,
# AND to correctly pass through any unexpected keyword arguments like 'task_type'.
class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        # The 'task_type' (or any other unexpected keyword argument) is passed by the
        # underlying LangChain code to this method. We need to accept it in kwargs
        # and pass it to the super method to prevent TypeError.
        raw_embeddings = super().embed_documents(texts, **kwargs) # Pass kwargs here

        processed_embeddings = []
        for single_embedding_raw in raw_embeddings:
            if isinstance(single_embedding_raw, (list, tuple)):
                if len(single_embedding_raw) == 1 and isinstance(single_embedding_raw[0], (list, tuple)):
                    processed_embeddings.append(list(single_embedding_raw[0]))
                else:
                    processed_embeddings.append(list(single_embedding_raw))
            elif hasattr(single_embedding_raw, '__iter__'): # Catches 'Repeated' objects specifically
                processed_embeddings.append(list(single_embedding_raw))
            else:
                raise TypeError(f"Expected iterable for single embedding, but received: {type(single_embedding_raw)}")
        
        return processed_embeddings

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        # This method in the base class will likely call self.embed_documents
        # passing 'task_type'. So we just need to ensure its output is flat
        # and also pass through any kwargs that the super method might expect.
        raw_embedding = super().embed_query(text, **kwargs) # Pass kwargs here as well
        
        if isinstance(raw_embedding, (list, tuple)):
            if len(raw_embedding) == 1 and isinstance(raw_embedding[0], (list, tuple)):
                return list(raw_embedding[0])
            else:
                return list(raw_embedding)
        elif hasattr(raw_embedding, '__iter__'): # Catches 'Repeated' objects specifically
            return list(raw_embedding)
        else:
            raise TypeError(f"Expected iterable for query embedding, but received: {type(raw_embedding)}")


# --- Flask App Initialization ---
app = Flask(__name__, template_folder='.')
load_dotenv()

# --- Configuration ---
VECTOR_DB_DIR = "chroma_db"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
GOOGLE_LLM_MODEL = "gemini-1.5-flash" # This model supports multimodal inputs

TASK_TYPE_DOCUMENT = "retrieval_document"
TASK_TYPE_QUERY = "retrieval_query"

# Configure Google Generative AI for direct multimodal calls
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
            task_type=TASK_TYPE_QUERY # Specify task_type for query embedding
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

with app.app_context():
    initialize_chatbot_components()

@app.route('/')
def serve_frontend():
    """Serves the main chatbot HTML page."""
    return render_template('index.html')

@app.route('/', methods=['POST']) # Allow POST to root for image uploads from promptfoo
def root_post_endpoint():
    return chat_endpoint() # Route all POST requests to chat_endpoint

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    API endpoint to receive a user question and return a chatbot answer.
    Expects a JSON payload: {"question": "Your question here?", "image": "optional_base64_image_data"}
    Returns a JSON payload: {"answer": "...", "links": [{"url": "...", "text": "..."}]}
    """
    user_data = request.get_json()

    if not user_data or 'question' not in user_data:
        return jsonify({"error": "Invalid request: Please provide a 'question' in the JSON body."}), 400

    user_question = user_data.get('question')
    base64_image = user_data.get('image') # Get base64 image data
    print(f"\nReceived API question: '{user_question}'")
    if base64_image:
        print("Image data received in request.")

    try:
        # Step 1: Embed the user's question (and potentially image for query)
        # For retrieval, we primarily use the text query for embeddings.
        # The image will be passed to the LLM directly.
        query_embedding = embeddings_model.embed_query(user_question)
        
        # Step 2: Perform similarity search using the query embedding
        retrieved_docs = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=4 # Retrieve 4 relevant documents for context
        )
        print(f"Found {len(retrieved_docs)} relevant documents.")

        # Step 3: Prepare the context and links for the LLM and response
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        links = []
        for doc in retrieved_docs:
            url = doc.metadata.get('url')
            # Prefer 'title' from Markdown, 'topic_title' from Discourse, or fallback to source/filename
            text = doc.metadata.get('title', doc.metadata.get('topic_title', doc.metadata.get('source', 'Relevant Document')))
            if url and text:
                links.append({"url": url, "text": text})

        if not context_text:
            context_text = "No specific relevant information found in the knowledge base."
            print("Warning: No context generated from retrieved documents.")

        # Step 4: Construct the prompt parts for the LLM (handling multimodal)
        prompt_parts = []
        if base64_image:
            try:
                # Decode base64 image
                image_data = base64.b64decode(base64_image)
                image_stream = BytesIO(image_data)
                img = Image.open(image_stream)
                
                # Add image to prompt parts
                prompt_parts.append(img)
                print("Successfully processed image for multimodal input.")

            except Exception as img_e:
                print(f"Error processing image: {img_e}")
                # Don't fail the entire request, just proceed without the image
                prompt_parts.append(f"Error processing image: {img_e}. Proceeding with text only.")

        # Add the main text query and context
        # The context and question are combined into a single prompt part.
        prompt_content = prompt.format(context=context_text, input=user_question)
        prompt_parts.append(prompt_content)
        
        # Step 5: Invoke the Multimodal LLM
        print("Invoking Multimodal LLM with prompt parts.")
        model_multimodal = genai.GenerativeModel(GOOGLE_LLM_MODEL) # Use direct genai client for multimodal
        
        # Use a timeout to ensure response within 30 seconds
        generation_config = {"temperature": 0.3}
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Max 25 seconds for LLM response, leaving 5 seconds for other processing
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
            # Set links to empty if LLM fails to prevent partial/malformed response
            links = [] 

        return jsonify({
            "answer": bot_answer,
            "links": links
        }), 200

    except Exception as e:
        print(f"Error during API chat processing: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e), "links": []}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    print("\n--- Starting Flask API Server ---")
    print("Frontend will be accessible at: http://127.0.0.1:5000/")
    print("API endpoint at: http://127.0.0.1:5000/chat (POST requests), also root '/' for promptfoo")
    app.run(host='0.0.0.0', port=5000, debug=True)
