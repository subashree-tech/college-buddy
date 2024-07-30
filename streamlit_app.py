import streamlit as st
from docx import Document
import openai
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Access your API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

INDEX_NAME = "college-buddy"

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536, 
        metric='cosine', 
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to truncate text
def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

# Function to count tokens
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to get embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def upsert_to_pinecone(text, file_name, file_id):
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]  # Split into 8000 character chunks
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        metadata = {
            "file_name": file_name,
            "file_id": file_id,
            "chunk_id": i,
            "chunk_text": chunk  # Make sure this line is present
        }
        index.upsert(vectors=[(f"{file_id}_{i}", embedding, metadata)])
        time.sleep(1)  # To avoid rate limiting
# Function to query Pinecone
def query_pinecone(query, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'chunk_text' in match['metadata']:
            contexts.append(match['metadata']['chunk_text'])
        else:
            # If 'chunk_text' is not in metadata, we'll use a placeholder
            contexts.append(f"Content from {match['metadata'].get('file_name', 'unknown file')}")
    return " ".join(contexts)

# Function to get answer from GPT-3.5-turbo
def get_answer(query):
    context = query_pinecone(query)
    max_context_tokens = 3000
    truncated_context = truncate_text(context, max_context_tokens)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant focused on answering questions based on the provided context."},
            {"role": "user", "content": f"Context: {truncated_context}\n\nQuestion: {query}"}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

# Streamlit Interface
st.set_page_config(page_title="College Buddy Assistant", layout="wide")
st.title("College Buddy Assistant")

# Sidebar for file upload and metadata
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload the Word Documents (DOCX)", type="docx", accept_multiple_files=True)
    if uploaded_files:
        total_token_count = 0
        for uploaded_file in uploaded_files:
            file_id = str(uuid.uuid4())
            transcript_text = extract_text_from_docx(uploaded_file)
            token_count = num_tokens_from_string(transcript_text)
            total_token_count += token_count
            
            # Upsert to Pinecone
            upsert_to_pinecone(transcript_text, uploaded_file.name, file_id)
            
            st.text(f"Uploaded: {uploaded_file.name}")
            st.text(f"File ID: {file_id}")
            
        st.subheader("Uploaded Documents")
        st.text(f"Total token count: {total_token_count}")

# Main content area
user_query = st.text_input("What would you like to know about the uploaded documents?")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Searching for the best answer..."):
            answer = get_answer(user_query)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question before searching.")

# Add a section for displaying recent questions and answers
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if user_query and 'answer' in locals():
    st.session_state.chat_history.append((user_query, answer))

if st.session_state.chat_history:
    st.header("Recent Questions and Answers")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):
        with st.expander(f"Q: {q}"):
            st.write(f"A: {a}")
