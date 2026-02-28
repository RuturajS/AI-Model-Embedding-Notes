from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# 1. Define the LLM Model
model = ChatOllama(model="mistral")

# 2. Retrieve and split the data from URLs
urls = ["YOUR_URL_HERE"] # Replace with your list of URLs
loader = WebBaseLoader(urls)
docs = loader.load()

# Split documents into chunks with an overlap of 100
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# 3. Convert to embeddings and store in Vector DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_collection"
)

# Set up the retriever
retriever = vector_db.as_retriever()

# --- BEFORE RAG ---
print("--- BEFORE RAG ---")
before_rag_template = "What is {topic}?"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Ollama"}))

# --- AFTER RAG ---
print("--- AFTER RAG ---")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model
    | StrOutputParser()
)

print(after_rag_chain.invoke("What is Ollama?"))
