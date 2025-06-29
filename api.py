import os
import traceback
from typing import List
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# --- Load Environment Variables ---
# This will automatically load OPENAI_API_KEY and OPENAI_BASE_URL from your .env file
load_dotenv()

# --- Configuration ---
VECTOR_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# --- Pydantic Models for a well-defined API structure ---
class ChatRequest(BaseModel):
    question: str
    image: str | None = None

class Link(BaseModel):
    url: str = Field(description="The exact URL of the source document.")
    text: str = Field(description="The title of the source document.")

class AnswerWithLinks(BaseModel):
    answer: str = Field(description="The comprehensive, concise answer to the user's question.")
    links: List[Link] = Field(description="A list of source links used to formulate the answer.")

# --- FastAPI App Initialization ---
app = FastAPI(title="TDS Virtual TA API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Global RAG Chain ---
rag_chain = None

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n---\n\n".join([f"Source URL: {doc.metadata.get('url')}\nSource Title: {doc.metadata.get('title')}\nContent: {doc.page_content}" for doc in docs])

@app.on_event("startup")
def initialize_rag_chain():
    """Initializes the RAG chain when the FastAPI app starts."""
    global rag_chain
    print("--- Initializing RAG Chain with AI Pipe (Standard Method) ---")
    
    # This is the simplified, standard, and correct way to initialize.
    # It will automatically use the keys from your .env file.
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)

    # Load the vector store and create a retriever
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Define the JSON output parser
    parser = JsonOutputParser(pydantic_object=AnswerWithLinks)
    
    # Define the prompt template with specific instructions
    template = """You are an expert Teaching Assistant for an IIT Madras AI course. Your personality is helpful, direct, and precise.
    Your goal is to provide a precise answer to the user's question based ONLY on the provided 'Context'.
    
    Carefully analyze the context to find numbers, scores, and rules. Perform calculations if necessary. 
    IMPORTANT FORMATTING RULE: When providing a final score, state it as a single number scaled out of 100 (e.g., "110", "95", "80"), not as a fraction (e.g., "11/10").
    
    You must format your response as a JSON object with two keys: "answer" and "links".
    The "answer" should be a concise response to the question.
    The "links" array must only contain the URLs and titles of the specific sources from the context that you used to formulate your answer.
    If the context is insufficient, the "answer" key should state "I do not have enough information to answer this question.", and the "links" array should be empty.

    CONTEXT:
    {context}
    
    FORMAT INSTRUCTIONS:
    {format_instructions}

    QUESTION:
    {question}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Construct the final RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    print("✅ RAG Chain Initialized Successfully.")

@app.post("/chat", response_model=AnswerWithLinks)
def chat_endpoint(request: ChatRequest):
    """Endpoint to handle chatbot questions."""
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain is not initialized.")
        
    try:
        user_question = request.question
        print(f"\nReceived question: '{user_question}'")
        if request.image:
            print("INFO: Image data received but will be ignored by this RAG chain.")
        
        print("⏳ Invoking RAG chain via AI Pipe...")
        response_data = rag_chain.invoke(user_question)
        
        print("✅ RAG chain executed successfully.")
        return response_data
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred.")
