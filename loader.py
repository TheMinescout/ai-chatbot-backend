# loader.py
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import re

load_dotenv()

# --- CONFIGURATION ---
# Read credentials from the environment (.env locally, or the host's secret store)
# instead of committing them to the repository.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "chatbot"
# The starting URL for your website. The scraper will follow links from here.
BASE_URL = "https://life.minescout.net"

# --- 1. INITIALIZE CONNECTIONS ---
print("🚀 Initializing connections...")
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Initialize the embedding model. This model is small, fast, and works well.
# It runs locally on your machine the first time you use it.
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Connections initialized.")

# --- 2. CONNECT TO YOUR PINECONE INDEX ---
print(f"Connecting to Pinecone index: '{INDEX_NAME}'...")
index = pc.Index(INDEX_NAME)
print("✅ Connected to index.")

# --- 3. WEB SCRAPING LOGIC ---
def scrape_website(url, visited_urls):
    """Recursively scrapes text from a website, following internal links."""
    if url in visited_urls or not url.startswith(BASE_URL):
        return "", set()

    print(f"🔎 Scraping: {url}")
    visited_urls.add(url)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    except requests.RequestException as e:
        print(f"❗️ Error fetching {url}: {e}")
        return "", set()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove script, style, nav, and footer tags to avoid clutter
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Get text and clean it up
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Remove excessive blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)

    # Find all links on the page that belong to the same domain
    links = set()
    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']
        # Properly join relative URLs
        if link.startswith('/'):
            link = requests.compat.urljoin(BASE_URL, link)
        if link.startswith(BASE_URL):
            links.add(link)

    return text, links

# --- 4. DATA PROCESSING AND UPLOADING ---
def process_and_upload():
    """Main function to run the scraping and uploading pipeline."""
    to_visit = {BASE_URL}
    visited = set()
    doc_id_counter = 0

    while to_visit:
        url = to_visit.pop()

        # Scrape the page
        page_text, new_links = scrape_website(url, visited)
        to_visit.update(new_links)

        if not page_text:
            continue

        # Split text into smaller, manageable chunks
        print(f"📝 Splitting text from {url} into chunks...")
        text_chunks = [page_text[i:i+1500] for i in range(0, len(page_text), 1500)]

        # Create embeddings for the chunks
        print(f"🧠 Creating embeddings for {len(text_chunks)} chunks...")
        embeddings = model.encode(text_chunks).tolist()

        # Prepare vectors for Pinecone format
        vectors_to_upsert = []
        for i, chunk in enumerate(text_chunks):
            vector_id = f"doc_{doc_id_counter}_{i}"
            vector = {
                "id": vector_id,
                "values": embeddings[i],
                "metadata": {"url": url, "text": chunk}
            }
            vectors_to_upsert.append(vector)

        # Upload the vectors to Pinecone
        if vectors_to_upsert:
            print(f"⏫ Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
            index.upsert(vectors=vectors_to_upsert)

        doc_id_counter += 1

    print("--- ✅ Scraping and uploading complete! ---")

# --- 5. SCRIPT EXECUTION ---
if __name__ == "__main__":
    user_input = input(f"❓ Are you sure you want to delete all vectors in the '{INDEX_NAME}' index? This is recommended for a fresh start. (yes/no): ")
    if user_input.lower() == 'yes':
        print("🗑️ Deleting all existing vectors in the index...")
        index.delete(delete_all=True)
        print("👍 Vectors deleted.")
        process_and_upload()
    else:
        print("🚫 Operation cancelled.")