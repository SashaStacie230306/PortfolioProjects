import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from datetime import datetime
import os

# --- Load test cases ---
with open("test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

# --- Load your documents ---
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
    path = os.path.join(base_path, file_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    content = json.dumps(item) if isinstance(item, dict) else str(item)
                    documents.append(Document(page_content=content, metadata={"source": file_name, "index": idx}))
            elif isinstance(data, dict):
                documents.append(Document(page_content=json.dumps(data), metadata={"source": file_name}))
            else:
                documents.append(Document(page_content=str(data), metadata={"source": file_name}))
    else:
        print(f"Warning: {file_name} not found.")

# --- Setup LangChain ---
llm = ChatOllama(model="llama3")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory="chroma_test_db")
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

# --- Run test cases ---
print("Running Marbet QA Tests...\n")

for case in test_cases:
    prompt = case["prompt"]
    expected_keywords = case["expected_keywords"]
    result = qa_chain({"question": prompt, "chat_history": []})
    response = result["answer"]

    hits = [kw for kw in expected_keywords if kw.lower() in response.lower()]
    score = len(hits) / len(expected_keywords) * 100

    print(f"Test {case['id']} ({case['category']})")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Keywords matched: {hits}")
    print(f"Score: {score:.0f}%\n")
