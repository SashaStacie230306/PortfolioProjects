from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load the local Ollama model
llm = ChatOllama(model="llama3.2:3b")

# Reconnect to Chroma vector store
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant for Marbet, a high-end event and travel management agency.
Your goal is to help guests by providing clear, accurate, and professional answers based only on the provided documents.
If the answer is not found in the documents, say: "I'm sorry, I couldn't find that information."

Context:
{context}

Guest Question:
{question}

Answer:
"""
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# Example question from a guest
query = "What documents do I need to bring on the trip?"
response = qa_chain.invoke(query)

# Output the final answer
print("\n Answer:")
print(response["result"])

# Optional: print the source docs used
print("\n Sources:")
for doc in response["source_documents"]:
    print(f"- {doc.metadata['source']}")
