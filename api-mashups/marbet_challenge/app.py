import streamlit as st
from datetime import datetime
import json
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- Configuration ---
st.set_page_config(page_title="Marbet Trip Assistant", layout="centered")

st.markdown("<div class='container'>", unsafe_allow_html=True)

if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

# Now safe to access
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0 if st.session_state["theme"] == "Light" else 1)
st.session_state["theme"] = theme


# --- Theme-Based Styling ---
light_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap');

body::before {
    content: "";
    background-image: url("C:/Users/sasha/Downloads/marbet-logo.jpg");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 200px;
    opacity: 0.04;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
}

html, body, [class*="css"]  {
    font-family: 'Open Sans', sans-serif;
    background-color: #F9F9F9;
    color: #2F2F2F;
}
.main-title {
    text-align: center;
    font-size: 36px;
    color: #FF3C00;
    font-weight: 600;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 2rem;
}
.chat-bubble {
    padding: 14px 18px;
    border-radius: 16px;
    margin: 12px 0;
    max-width: 85%;
    word-wrap: break-word;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
.chat-container {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 1rem;
}
.you {
    background-color: #ECECEC;
    text-align: right;
    margin-left: auto;
}
.marbet {
    background-color: white;
    border-left: 4px solid #FF3C00;
    margin-right: auto;
    text-align: left;
}
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 14px;
    color: #888;
}
.footer a {
    color: #FF3C00;
    text-decoration: none;
}
</style>
"""

dark_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap');

body::before {
    content: "";
    background-image: url("C:/Users/sasha/Downloads/marbet-logo.jpg");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 200px;
    opacity: 0.04;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
}

html, body, [class*="css"]  {
    font-family: 'Open Sans', sans-serif;
    background-color: #121212;
    color: #F0F0F0;
}
.main-title {
    text-align: center;
    font-size: 36px;
    color: #FF3C00;
    font-weight: 600;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #BBBBBB;
    margin-bottom: 2rem;
}
.chat-bubble {
    padding: 14px 18px;
    border-radius: 16px;
    margin: 12px 0;
    max-width: 85%;
    word-wrap: break-word;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
.chat-container {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 1rem;
}
.you {
    background-color: #2D2D2D;
}
.marbet {
     background-color: #202020;
    border-left: 4px solid #FF3C00;
}
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 14px;
    color: #BBB;
}
.footer a {
    color: #FFA63D;
    text-decoration: none;
}
</style>
"""

# Apply CSS
st.markdown(light_css if st.session_state["theme"] == "Light" else dark_css, unsafe_allow_html=True)

# --- Logo ---
image_path = "C:/Users/sasha/Downloads/marbet-logo.jpg"
if os.path.exists(image_path):
    st.image(image_path, width=240)
else:
    st.warning("Marbet logo not found!")

# --- Load and Process JSON Files ---
base_path = "C:/Users/sasha/Documents/GitHub/week9_10/marbet_challenge/challenge"
file_paths = [
    "esta_info.json",
    "eta_info.json",
    "extracted_activities_v2.json",
    "packing_list.json",
    "scenic_eclipse_cleaned.json",
    "spa_services.json",
    "wifi_setup_steps.json"
]

documents = []
for file_name in file_paths:
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        content = json.dumps(item) if isinstance(item, dict) else str(item)
                        documents.append(Document(page_content=content, metadata={"source": file_name, "index": idx}))
                elif isinstance(data, dict):
                    content = json.dumps(data)
                    documents.append(Document(page_content=content, metadata={"source": file_name}))
                else:
                    content = str(data)
                    documents.append(Document(page_content=content, metadata={"source": file_name}))
            except json.JSONDecodeError:
                st.warning(f"Failed to parse {file_name}. Ensure it's valid JSON.")
    else:
        st.warning(f"File not found: {file_name}")

# --- LangChain Setup ---
llm = ChatOllama(model="llama3")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory="chroma_db")
retriever = vectorstore.as_retriever()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a friendly and knowledgeable travel AI event assistant for an incentive trip for Marbet, a premium event and travel agency. 
Your goal is to help guests with practical and easy-to-understand answers based on the documents provided. 
Use a welcoming and conversational tone. Keep answers clear and concise, and speak like a helpful human, not a robot.

If you can’t find the answer in the documents, say: 
"I'm not sure about that based on the trip details I have, but I recommend checking with your Marbet contact."

Context:
{context}

Guest Question:
{question}

Answer:
"""
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)


# --- Suggested Questions ---
suggestions = [
    "What should I pack?",
    "What’s the schedule on Monday?",
    "Is WiFi available on board?",
    "Do I need a visa?",
]

# --- Suggestion + Clear Chat ---
col1, col2 = st.columns([3, 1])
with col1:
    suggestion = st.selectbox("Suggested questions:", [""] + suggestions)
with col2:
    if st.button("Clear chat"):
        st.session_state["chat_history"] = []

# --- Input Handling ---
user_input = st.chat_input("Type your question here...")
if not user_input and suggestion:
    user_input = suggestion


# --- Custom logic for known issues (e.g. visa) ---
def custom_logic(question):
    if "visa" in question.lower():
        return (
            "Most guests will require either an ESTA (for U.S. citizens) or an ETA (for Canadian and other nationalities). "
            "Please ensure your passport is valid and you check your eligibility with official immigration websites. "
            "Contact Marbet if you're unsure."
        )
    return None

def humanize_response(text):
    # Remove boilerplate phrases
    replacements = [
        ("As an intelligent and professional assistant", ""),
        ("Based on the information provided,", ""),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text.strip().lstrip(",")

# --- Run query ---
if user_input:
    with st.spinner("Thinking..."):
        fallback = custom_logic(user_input)
        if fallback:
            answer = fallback
        else:
            chat_history_only = [(q, a) for q, a in st.session_state.chat_history if q != "Marbet"]
            result = qa_chain({"question": user_input, "chat_history": chat_history_only})
            answer = result["answer"]
        # Log query to JSONL
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": user_input,
            "answer": answer
        }

        log_file = "query_logs.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update chat
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Marbet", answer))

        answer = humanize_response(answer)


# Header
st.markdown("""
    <div class="main-title">Marbet Trip Assistant</div>
    <div class="sub-title">Your personal event and travel helper</div>
""", unsafe_allow_html=True)

# --- Chat Display ---

st.markdown('<div class="chat-container">', unsafe_allow_html=True)


for speaker, message in st.session_state.chat_history:
    role = "you" if speaker == "You" else "marbet"
    st.markdown(f"""
        <div class='chat-bubble {role}'>
            <strong>{speaker}:</strong><br>{message}
        </div>
    """, unsafe_allow_html=True)


# --- Sticky Footer ---
st.markdown("""
    <div class="footer">
        © 2025 Marbet. All rights reserved. | <a href="https://www.marbet.com/en" target="_blank">www.marbet.com</a>
    </div>
""", unsafe_allow_html=True)




