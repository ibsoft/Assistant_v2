from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import ollama
from langchain_community import embeddings

# Initialize the local model
model_local = ChatOllama(model="tinydolphin")

# Define PDF paths
pdf_paths = [
    "Venus_The_Atmosphere_Climate_Surface_Interior_and_.pdf"
]

# Initialize Chroma vector store
vectorstore = Chroma(collection_name="rag-chroma")

# 1. Load PDF files from disk and split data into chunks
docs = [PyPDFLoader(path).load() for path in pdf_paths]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='all-minilm'),
)
retriever = vectorstore.as_retriever()



# 3. RAG
print("\n########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("What is Venus?"))

# Chat loop
print("\nWelcome to the Chatbot! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = after_rag_chain.invoke(user_input)
    print("Bot:", response)
