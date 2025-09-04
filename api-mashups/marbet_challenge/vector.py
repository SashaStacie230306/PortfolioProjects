from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
from langchain.chains import RetrievalQA

# Define paths
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

# Helper functions to parse JSON files
def parse_esta(data):
    combined = []
    combined.append(f"{data['title']}. {data['overview']}")
    combined.append("Requirements:\n" + "\n".join(f"- {r}" for r in data["requirements"]))
    combined.append("Instructions:\n" + "\n".join(f"{k}: {v}" for k, v in data["instructions"].items()))
    combined.append("Additional Info:\n" + "\n".join(f"{k}: {v}" for k, v in data["additional_info"].items()))
    combined.append(f"Contact Note: {data.get('contact_note', '')}")
    combined.append("Status Check:\n" + "\n".join(data["status_check_instructions"].get("actions", [])))
    combined.append("TIP: " + data.get("TIP", ""))
    return "\n\n".join(combined)

def parse_eta(data):
    combined = []
    combined.append(f"{data['title']}. {data['overview']}")
    combined.append("Requirements:\n" + "\n".join(f"- {r}" for r in data["requirements"]))
    combined.append("Instructions:\n" + "\n".join(f"{k}: {v}" for k, v in data["instructions"].items()))
    if "important_notes" in data:
        combined.append("Important Notes:\n" + "\n".join(f"{k}: {v}" for k, v in data["important_notes"].items()))
    if "confirmation" in data:
        combined.append("Confirmation Info:\n" + "\n".join(f"{k}: {v}" for k, v in data["confirmation"].items()))
    return "\n\n".join(combined)

# Load and process documents
docs = []
for filename in file_paths:
    full_path = os.path.join(base_path, filename)
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "esta" in filename:
            content = parse_esta(data)
        elif "eta" in filename:
            content = parse_eta(data)
        else:
            content = json.dumps(data, indent=2)
        docs.append(Document(page_content=content, metadata={"source": filename}))

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

# Initialize embedding model
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Load the existing Chroma vector store
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# Convert the vector store into a retriever
retriever = vectorstore.as_retriever()

# Initialize the language model
llm = ChatOllama(model="llama3.2:3b")

# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

print("Vector store built and saved to 'chroma_db'")
