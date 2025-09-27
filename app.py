# app.py
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# --- INITIALIZATION ---
# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes, allowing your website to talk to this API
CORS(app)

# Hugging Face and Pinecone credentials
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "chatbot"

# We will use a powerful open-source model from Mistral AI via the Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(INDEX_NAME)

# Initialize the sentence embedding model (same as in loader.py)
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- HELPER FUNCTIONS ---
def query_huggingface_api(payload):
    """Sends a query to the Hugging Face Inference API and gets a response."""
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status() # Raise an exception for bad status codes
    return response.json()

# --- API ROUTE ---
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Get the user's question from the request
        data = request.get_json()
        user_question = data.get('question')

        if not user_question:
            return jsonify({"error": "No question provided"}), 400

        # 1. Create an embedding for the user's question
        question_embedding = model.encode(user_question).tolist()

        # 2. Query Pinecone to find the most relevant context
        query_response = index.query(
            vector=question_embedding,
            top_k=3, # Get the top 3 most relevant text chunks
            include_metadata=True
        )

        # 3. Assemble the context from the Pinecone results
        context = ""
        if query_response['matches']:
            for match in query_response['matches']:
                context += match['metadata']['text'] + "\n\n"

        # If no context is found, provide a default response
        if not context:
             return jsonify({"answer": "I'm sorry, I couldn't find any information related to your question on this website. Please try asking something else."})

        # 4. Create the prompt for the AI model
        # This is a crucial step. We instruct the model on how to behave.
        prompt = f"""
        You are a helpful AI assistant for a website.
        Your role is to answer user questions based ONLY on the context provided below.
        If the context does not contain the answer, you MUST say "I'm sorry, I couldn't find information about that on this website."
        Do not make up answers. Be concise and helpful.

        CONTEXT:
        {context}

        USER'S QUESTION:
        {user_question}

        YOUR ANSWER:
        """

        # 5. Query the AI model
        output = query_huggingface_api({
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 250, # Limit the length of the response
                "return_full_text": False # Only return the generated part
            }
        })

        # The API returns a list, so we get the first element
        ai_response = output[0]['generated_text']

        # 6. Send the response back to the user
        return jsonify({"answer": ai_response})

    except Exception as e:
        # Handle potential errors gracefully
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Run the app. `debug=True` allows for automatic reloading when you save the file.
    # Remove `debug=True` when you deploy to production.
    app.run(host='0.0.0.0', port=5000, debug=True)