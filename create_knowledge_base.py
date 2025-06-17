import os
import json
import shutil
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings # IMPORTANT: Using the OpenAI library
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from datetime import datetime
from dotenv import load_dotenv

# This will load OPENAI_API_KEY and OPENAI_BASE_URL from your .env file
load_dotenv() 

# --- Configuration ---
DISCOURSE_CLEANED_FILE = "discourse_posts_cleaned.json"
MARKDOWN_DIR = "markdown_files"
VECTOR_DB_DIR = "chroma_db"
# This model name is compatible with the AI Pipe proxy
EMBEDDING_MODEL = "text-embedding-3-small"

# Note: We no longer need the CustomGoogleGenerativeAIEmbeddings class

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

    markdown_documents = []
    for doc in docs:
        content_lines = doc.page_content.split('\n')
        frontmatter = {}
        main_content = doc.page_content

        if content_lines and content_lines[0].strip() == '---':
            # This logic parses metadata from the top of the markdown file
            frontmatter_started = False
            content_start_line = 0
            for i, line in enumerate(content_lines):
                if line.strip() == '---':
                    if not frontmatter_started:
                        frontmatter_started = True
                    else:
                        content_start_line = i + 1
                        break
                elif frontmatter_started and ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip().strip('"')
            main_content = "\n".join(content_lines[content_start_line:]).strip()

        if not main_content:
            continue

        processed_doc = Document(
            page_content=main_content,
            metadata={
                "source": "course_material",
                "url": frontmatter.get('original_url', doc.metadata.get('source', '')),
                "title": frontmatter.get('title', os.path.basename(doc.metadata.get('source', '')).replace(".md", "")),
                "downloaded_at": frontmatter.get('downloaded_at', datetime.now().isoformat())
            }
        )
        markdown_documents.append(processed_doc)

    print(f"✅ Generated {len(markdown_documents)} initial Markdown documents.")
    return markdown_documents

def create_and_store_knowledge_base():
    """Main function to build and store the vector database."""
    print("--- Creating Knowledge Base using AI Pipe Proxy ---")
    
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
    # It will AUTOMATICALLY use the OPENAI_API_KEY and OPENAI_BASE_URL from your .env file.
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