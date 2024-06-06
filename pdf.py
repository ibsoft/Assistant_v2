from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import langchain_community.embeddings.ollama as ollama
import os

# Initialize the local model
model_local = ChatOllama(model="tinydolphin")

# ChromaDB settings
collection_name = "rag-chroma"
persist_directory = "chromadb_storage"

# 1. Load PDF files from disk and split data into chunks
pdf_paths = ["CyBOK-version-1.0.pdf"]

# Initialize ChromaDB
vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory
)

# If the collection is empty, load and process the PDF
if vectorstore.is_empty():
    print("Loading and processing PDF files...")
    docs = [PyPDFLoader(path).load() for path in pdf_paths]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Create embeddings
    embedding_model = ollama.OllamaEmbeddings(model='all-minilm')
    embeddings = [embedding_model.embed(document.text) for document in doc_splits]
    
    # Store the processed documents in ChromaDB
    vectorstore.add_documents(doc_splits, embeddings)
    vectorstore.persist()
else:
    print("Loading documents from ChromaDB...")

retriever = vectorstore.as_retriever()

# 2. Define RAG prompt
rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | model_local
    | StrOutputParser()
)

# 3. Start the chat loop
print("Welcome to the PDF Chatbot! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = rag_chain.invoke(user_input)
    print(f"Bot: {response}")
