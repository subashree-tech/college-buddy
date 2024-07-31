import streamlit as st
from docx import Document
import openai
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
from openai import OpenAI
# Access your API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "college-buddy"
# Initialize OpenAI
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
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
# List of example questions
EXAMPLE_QUESTIONS = [
    "What are the steps to declare a major at Texas Tech University",
    "What are the GPA and course requirements for declaring a major in the Rawls College of Business?",
    "How can new students register for the Red Raider Orientation (RRO)",
    "What are the key components of the Texas Tech University Code of Student Conduct",
    "What resources are available for students reporting incidents of misconduct at Texas Tech University",
    "What are the guidelines for amnesty provisions under the Texas Tech University Code of Student Conduct",
    "How does Texas Tech University handle academic misconduct, including plagiarism and cheating",
    "What are the procedures for resolving student misconduct through voluntary resolution or formal hearings",
    "What are the rights and responsibilities of students during the investigative process for misconduct at Texas Tech University",
    "How can students maintain a healthy lifestyle, including nutrition and fitness, while attending Texas Tech University"
]
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
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
            )
    return response.data[0].embedding
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
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are College Buddy, an advanced AI assistant designed to help students with their academic queries. Your primary function is to analyze and provide insights based on the context of uploaded documents. Please adhere to the following guidelines:
1. Focus on delivering accurate, relevant information derived from the provided context.
2. If the context doesn't contain sufficient information to answer a query, state this clearly and offer to help with what is available.
3. Maintain a friendly, supportive tone appropriate for assisting students.
4. Provide concise yet comprehensive answers, breaking down complex concepts when necessary.
5. If asked about topics beyond the scope of the provided context, politely redirect the conversation to the available information.
6. Encourage critical thinking by guiding students towards understanding rather than simply providing direct answers.
7. Respect academic integrity by not writing essays or completing assignments on behalf of students.
8. Additional Resources: Suggest any additional resources or videos for further learning.
   Include citations for the title of the video the information is from and the timestamp where relevant information is presented.
9. When mentioning or referring to any of the following tools or services, always include the corresponding link immediately after the name:
   - Raiderlink: https://raiderlink.ttu.edu
   - DegreeWorks: https://appserv.itts.ttu.edu/DegreeWorks/Dashboard
   - Schedule Builder: https://tturedraider.collegescheduler.com
   - Blackboard: https://ttu.blackboard.com
   - Texas Tech Email: https://outlook.office365.com
   - Library Website: https://www.depts.ttu.edu/library/
   - Student Business Services: https://www.depts.ttu.edu/studentbusinessservices/
   - Academic Catalog: https://catalog.ttu.edu/
   For example, if discussing Raiderlink, write it as: Raiderlink (https://raiderlink.ttu.edu)
10. If a student asks about any of these tools or services without specifically mentioning the name, recognize the context and provide the name along with the link.
11. Ensure that every mention of these tools or services includes the link, even if mentioned multiple times in the same response.
Your goal is to be a knowledgeable, reliable, and supportive academic companion, enhancing the learning experience of the students you assist. Always provide easy access to these tools and services through consistent linking."""},
            {"role": "user", "content": f"Context: {truncated_context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content.strip()
# Streamlit Interface
st.set_page_config(page_title="College Buddy Assistant", layout="wide")
st.title("College Buddy Assistant")
st.markdown("Welcome to College Buddy! I am here to help you stay organized, find information fast and provide assistance. Feel free to ask me a question below.")
# Sidebar for file upload and metadata
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload the Word Documents (DOCX)", type="docx", accept_multiple_files=True)
    if uploaded_files:
        total_token_count = 0
        for uploaded_file in uploaded_files:
            file_id = str(uuid.uuid4())
            text = extract_text_from_docx(uploaded_file)
            token_count = num_tokens_from_string(text)
            total_token_count += token_count
            # Upsert to Pinecone
            upsert_to_pinecone(text, uploaded_file.name, file_id)
            st.text(f"Uploaded: {uploaded_file.name}")
            st.text(f"File ID: {file_id}")
        st.subheader("Uploaded Documents")
        st.text(f"Total token count: {total_token_count}")
# Main content area
st.header("Popular Questions")
# Randomly select 3 questions
selected_questions = random.sample(EXAMPLE_QUESTIONS, 3)
for question in selected_questions:
    if st.button(question):
        get_answer(question)
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
