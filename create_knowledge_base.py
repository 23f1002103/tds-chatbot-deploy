import os
import json
import shutil
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from datetime import datetime
from dotenv import load_dotenv

# This will automatically load OPENAI_API_KEY and OPENAI_BASE_URL
load_dotenv() 

# --- Configuration ---
DISCOURSE_CLEANED_FILE = "discourse_posts_cleaned.json"
MARKDOWN_DIR = "markdown_files"
VECTOR_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Helper functions (no changes needed) ---
def clean_text(text: str) -> str:
    """Basic text cleaning for consistency."""
    if text is None:
        return ""
    return " ".join(text.strip().split())

def load_discourse_documents(file_path: str) -> list[Document]:
    """Loads cleaned Discourse posts and creates a separate LangChain Document for EACH post."""
    print(f"Loading and processing each Discourse post from {file_path} individually...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            posts_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure scraper has run.")
        return []
    
    discourse_documents = []
    for post in tqdm(posts_data, desc="Processing individual Discourse posts"):
        topic_title = post.get("topic_title", "")
        post_content = clean_text(post.get("content", ""))
        page_content = f"Forum Topic: {topic_title}\n\nUser '{post.get('username', 'N/A')}' wrote:\n{post_content}"
        
        post_url = post.get('url', '')
        if not post_url: # Construct URL if it's missing
            post_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{post.get('topic_slug', '')}/{post.get('topic_id', '')}/{post.get('post_number', '')}"
        
        metadata = {
            "source": "discourse_forum", "url": post_url, "title": topic_title,
            "topic_id": post.get("topic_id"), "post_number": post.get("post_number"),
            "username": post.get("username"), "created_at": post.get("created_at"), "updated_at": post.get("updated_at")
        }
        discourse_documents.append(Document(page_content=page_content, metadata=metadata))
        
    print(f"✅ Generated {len(discourse_documents)} documents from {len(posts_data)} Discourse posts.")
    return discourse_documents

def load_markdown_documents(directory: str) -> list[Document]:
    """Loads Markdown files and processes their frontmatter."""
    print(f"Loading Markdown files from {directory} into initial documents...")
    try:
        loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
        )
        docs = loader.load()
    except Exception as e:
        print(f"Error loading Markdown files from {directory}: {e}")
        return []

    print(f"Loaded {len(docs)} raw Markdown documents.")
    # You can add your frontmatter processing here if needed
    return docs

def create_and_store_knowledge_base():
    """Main function to build and store the vector database."""
    print("--- Creating Knowledge Base using AI Pipe Proxy (Standard Method) ---")
    
    # 1. Load all documents
    all_raw_documents = load_discourse_documents(DISCOURSE_CLEANED_FILE) + load_markdown_documents(MARKDOWN_DIR)

    if not all_raw_documents:
        print("No documents were loaded. Exiting.")
        return

    # 2. Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    all_chunks = text_splitter.split_documents(all_raw_documents)
    print(f"✅ Generated {len(all_chunks)} final chunks for embedding.")

    # 3. Initialize the OpenAI Embeddings model. 
    # This simplified call will AUTOMATICALLY use the environment variables.
    print(f"⏳ Initializing OpenAI embedding model '{EMBEDDING_MODEL}' via AI Pipe...")
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    print("✅ OpenAI Embedding model loaded successfully.")

    # 4. Delete old database and create a new one
    if os.path.exists(VECTOR_DB_DIR):
        print(f"⚠️ Deleting existing ChromaDB directory: {VECTOR_DB_DIR}")
        shutil.rmtree(VECTOR_DB_DIR)

    print(f"⏳ Creating new ChromaDB and storing {len(all_chunks)} chunks...")
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings_model,
        persist_directory=VECTOR_DB_DIR
    )
    print("✅ Vector database created successfully.")

if __name__ == "__main__":
    create_and_store_knowledge_base()

