import os
import json
import re
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any
from tqdm import tqdm
import shutil
import traceback # Import traceback for detailed error logging

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

# --- Custom Embedding Wrapper (Keep this exactly as is) ---
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


# --- Configuration ---
DISCOURSE_CLEANED_FILE = "discourse_posts_cleaned.json"
MARKDOWN_DIR = "markdown_files"
VECTOR_DB_DIR = "chroma_db"

GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
TASK_TYPE_DOCUMENT = "retrieval_document"
TASK_TYPE_QUERY = "retrieval_query"

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")

# --- Helper functions for Discourse subthread processing (unchanged) ---
def clean_text(text):
    """Basic text cleaning for consistency."""
    if text is None:
        return ""
    return " ".join(text.strip().split())

def build_reply_map(posts):
    """
    Builds a map: parent_post_number -> list of child posts
    """
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        if "post_number" not in post or "topic_id" not in post:
            print(f"Warning: Skipping malformed post due to missing essential keys (post_number/topic_id): {post.get('post_id', 'N/A')}")
            continue
        posts_by_number[post["post_number"]] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    """
    Recursively collect all posts in a subthread rooted at root_post_number
    """
    collected = []
    def dfs(post_num):
        post = posts_by_number.get(post_num)
        if not post:
            return
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

# --- Document Loading (Initial Documents - NOT yet chunked) (unchanged) ---

def load_discourse_documents(file_path: str) -> List[Document]:
    """
    Loads cleaned Discourse posts and combines them into initial LangChain Document objects
    where each document represents a full subthread. These will be further chunked later.
    """
    print(f"Loading Discourse posts from {file_path} into initial documents...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            posts_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the discourse scraper and cleaner have run.")
        return []
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file. Please check its content.")
        return []

    if not posts_data:
        print(f"Warning: {file_path} is empty or contains no posts.")
        return []

    topics = {}
    for post in posts_data:
        topic_id = post.get("topic_id")
        if topic_id is None:
            print(f"Warning: Skipping post {post.get('post_id', 'N/A')} due to missing topic_id.")
            continue
        if topic_id not in topics:
            topics[topic_id] = {"topic_title": post.get("topic_title", ""), "posts": []}
        topics[topic_id]["posts"].append(post)

    for topic_id in topics:
        topics[topic_id]["posts"].sort(key=lambda p: p.get("post_number", 0))

    print(f"Loaded {len(posts_data)} posts across {len(topics)} topics from Discourse.")

    discourse_documents = []
    for topic_id, topic_data in tqdm(topics.items(), desc="Combining Discourse subthreads"):
        posts = topic_data["posts"]
        topic_title = topic_data["topic_title"]

        if not posts:
            continue

        reply_map, posts_by_number = build_reply_map(posts)
        all_post_numbers = set(p["post_number"] for p in posts)
        actual_root_posts = [
            p for p in posts
            if p.get("reply_to_post_number") is None
            or p.get("reply_to_post_number") not in all_post_numbers
        ]

        for root_post in actual_root_posts:
            root_num = root_post["post_number"]
            subthread_posts = extract_subthread(root_num, reply_map, posts_by_number)

            if not subthread_posts:
                continue

            combined_text = f"Topic title: {topic_title}\n\n"
            combined_text += "\n\n---\n\n".join(
                clean_text(p["content"]) for p in subthread_posts
            )

            first_post_created_at = subthread_posts[0].get('created_at', '')
            if first_post_created_at:
                try:
                    first_post_created_at = datetime.fromisoformat(first_post_created_at.replace('Z', '+00:00')).isoformat()
                except ValueError:
                    first_post_created_at = ''

            last_post_updated_at = subthread_posts[-1].get('updated_at', '')
            if last_post_updated_at:
                try:
                    last_post_updated_at = datetime.fromisoformat(last_post_updated_at.replace('Z', '+00:00')).isoformat()
                except ValueError:
                    last_post_updated_at = ''

            metadata = {
                "source": "discourse_forum",
                "url": root_post.get('url', ''),
                "topic_id": topic_id,
                "topic_title": topic_title,
                "root_post_number": root_num,
                "first_post_created_at": first_post_created_at,
                "last_post_updated_at": last_post_updated_at
            }
            metadata["post_ids_in_subthread"] = ",".join(map(str, [p.get("post_id", "N/A") for p in subthread_posts]))
            metadata["post_numbers_in_subthread"] = ",".join(map(str, [p.get("post_number", "N/A") for p in subthread_posts]))

            discourse_documents.append(Document(page_content=combined_text, metadata=metadata))

    print(f"✅ Generated {len(discourse_documents)} initial Discourse documents (subthreads).")
    return discourse_documents

def load_markdown_documents(directory: str) -> List[Document]:
    """
    Loads Markdown files into initial LangChain Document objects, extracting frontmatter.
    These will be further chunked later.
    """
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

    if not docs:
        print(f"Warning: No Markdown files found or loaded from {directory}.")
        return []

    print(f"Loaded {len(docs)} raw Markdown documents.")

    markdown_documents = []
    for doc in docs:
        content_lines = doc.page_content.split('\n')
        frontmatter = {}
        main_content = doc.page_content # Default to full content if no frontmatter

        if content_lines and content_lines[0].strip() == '---':
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
            print(f"Warning: Markdown file {doc.metadata.get('source', 'unknown')} has empty content after frontmatter removal.")
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

# --- Main Process ---
def create_and_store_knowledge_base():
    print("--- Initializing Knowledge Base Creation ---")

    # Define consistent chunking parameters
    GLOBAL_CHUNK_SIZE = 400
    GLOBAL_CHUNK_OVERLAP = 80

    # Load initial documents from both sources
    discourse_docs_full = load_discourse_documents(DISCOURSE_CLEANED_FILE)
    markdown_docs_full = load_markdown_documents(MARKDOWN_DIR)

    # --- REMOVED: TEMPORARY LIMIT DATASET SIZE FOR LOCAL TESTING ---
    # discourse_docs = discourse_docs_full[:5]
    # markdown_docs = markdown_docs_full[:5]
    # print(f"DEBUG: Temporarily limiting Discourse documents to {len(discourse_docs)} for testing.")
    # print(f"DEBUG: Temporarily limiting Markdown documents to {len(markdown_docs)} for testing.")
    # --- END REMOVED ---

    all_raw_documents = discourse_docs_full + markdown_docs_full # Use full datasets
    if not all_raw_documents:
        print("No documents loaded from any source. Exiting.")
        return
    print(f"Total raw documents loaded (before splitting): {len(all_raw_documents)}")

    # --- Apply consistent chunking to ALL raw documents ---
    print(f"⏳ Splitting all documents into chunks using RecursiveCharacterTextSplitter (chunk_size={GLOBAL_CHUNK_SIZE}, chunk_overlap={GLOBAL_CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=GLOBAL_CHUNK_SIZE,
        chunk_overlap=GLOBAL_CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    all_chunks = text_splitter.split_documents(all_raw_documents)
    if not all_chunks:
        print("No chunks generated after splitting. Exiting.")
        return

    print(f"✅ Generated {len(all_chunks)} final chunks for embedding.")

    # --- Embedding Generation (using the Custom Wrapper) ---
    print(f"⏳ Initializing Custom Google embedding model: {GOOGLE_EMBEDDING_MODEL} with task_type='{TASK_TYPE_DOCUMENT}'...")
    try:
        embeddings_model = CustomGoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL, 
            task_type=TASK_TYPE_DOCUMENT # Specify task_type for document embedding
        )
    except Exception as e:
        print(f"Error initializing Custom Google Embedding model: {e}. Check your GOOGLE_API_KEY and internet connection. Exiting.")
        traceback.print_exc()
        return
    print("✅ Custom Google Embedding model loaded.")

    # --- Vector Database Storage (ChromaDB) ---
    print(f"⏳ Creating or loading ChromaDB at {VECTOR_DB_DIR} and storing embeddings...")

    # Delete existing ChromaDB directory to ensure a fresh start with new chunks
    if os.path.exists(VECTOR_DB_DIR):
        print(f"⚠️ Deleting existing ChromaDB directory: {VECTOR_DB_DIR}")
        try:
            shutil.rmtree(VECTOR_DB_DIR)
            print("Successfully deleted old ChromaDB.")
        except OSError as e:
            print(f"Error deleting old ChromaDB: {e}. Please delete it manually if issues persist.")
            traceback.print_exc()

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    BATCH_SIZE = 50 # Google API limit is 100, so use slightly less to be safe
    print(f"⏳ Storing {len(all_chunks)} chunks in ChromaDB in batches of {BATCH_SIZE}...")

    vectorstore = None
    try:
        if not all_chunks:
            print("No chunks to add to ChromaDB. Skipping vectorstore creation.")
            return

        print(f"Creating ChromaDB with first batch of {len(all_chunks[0:BATCH_SIZE])} documents.")
        vectorstore = Chroma.from_documents(
            documents=all_chunks[0:BATCH_SIZE],
            embedding=embeddings_model,
            persist_directory=VECTOR_DB_DIR
        )

        for i in tqdm(range(BATCH_SIZE, len(all_chunks), BATCH_SIZE), desc="Adding remaining chunks to ChromaDB"):
            batch = all_chunks[i:i + BATCH_SIZE]
            if batch:
                vectorstore.add_documents(documents=batch)

        vectorstore.persist()
        print("✅ Vector database created and saved successfully.")
    except Exception as e:
        print(f"\nError during ChromaDB creation/persistence: {e}")
        traceback.print_exc()
        return

    print("\nKnowledge base creation complete! You can now use this vector database for retrieval.")

if __name__ == "__main__":
    create_and_store_knowledge_base()

