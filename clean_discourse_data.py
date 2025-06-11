import json
import re # In case you need regex for more advanced PII removal

INPUT_FILE = "discourse_posts.json"
OUTPUT_FILE = "discourse_posts_cleaned.json" # Save to a new file to keep original

def clean_post_data(post_data):
    """
    Removes or anonymizes PII from a single post dictionary.
    """
    # Anonymize author usernames/IDs
    if 'author' in post_data:
        # You can make this smarter if you know specific role patterns
        # E.g., if 'carlton' is an instructor, replace with 'Instructor'
        # For student IDs, replace with generic 'Student'
        if re.match(r'^\d{10}$', post_data['author']): # Simple regex for 10-digit IDs
            post_data['author'] = "Student"
        elif post_data['author'].lower() in ['carlton', 'adityach1']: # Example, replace with known TA/Instructor names
            post_data['author'] = "Teaching Assistant" # Or "Instructor" if applicable
        else:
            post_data['author'] = "Anonymous User" # Default for other usernames

    # Clear mentioned users
    if 'mentioned_users' in post_data:
        post_data['mentioned_users'] = []

    # Optional: Redact potential PII from content (e.g., emails, phone numbers if they appear)
    # This is more advanced and might require careful regex and testing.
    # For a course project, focusing on explicit author/mentioned_users might be enough.
    # if 'content' in post_data and post_data['content'] is not None:
    #     # Example: Redact emails
    #     post_data['content'] = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', post_data['content'])
    #     # Example: Redact 10-digit phone numbers (simple, might catch other numbers)
    #     post_data['content'] = re.sub(r'\b\d{10}\b', '[PHONE_REDACTED]', post_data['content'])


    return post_data

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            posts = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please ensure the discourse scraper has run.")
        return

    print(f"Cleaning {len(posts)} posts...")
    cleaned_posts = [clean_post_data(post) for post in posts]

    print(f"Saving cleaned data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_posts, f, indent=2)

    print(f"✅ PII removal complete. Cleaned data saved to {OUTPUT_FILE}")
    print(f"Please update your RAG ingestion process to use {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()