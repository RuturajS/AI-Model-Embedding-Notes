Based on the video transcript, the tutorial walks through building a local, zero-cost Retrieval-Augmented Generation (RAG) application. It extracts data from URLs, converts the data into embeddings using Nomic, stores them in Chroma DB, and uses the Mistral large language model via Ollama to answer questions based on that data ****. Finally, it wraps the application in a Gradio user interface ****.

Because the source is a spoken transcript rather than raw code, I have reconstructed the Python code based on the exact LangChain components, models, and steps mentioned in the video ****.

### Prerequisites & Context
Before running the code, you need to set up your environment and download the necessary models locally using Ollama:
1. **Install Python dependencies:** 
   `pip install langchain langchain-community langchain-core` **** *(Note: you will also need to install `chromadb` and `gradio` to run the UI version)*.
2. **Download local models via Ollama:**
   Run `ollama pull nomic-embed-text` to get the embedding model ****.
   Run `ollama pull mistral` to get the large language model ****.

### 1. Command-Line Version (`app.py`)
This script compares how the Mistral model answers a question *before* RAG (relying only on its training data) and *after* RAG (using the context from the provided URLs) ****.

```python
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
```

### 2. Gradio User Interface Version (`ui.py`)
The creator later modifies the code to add a Gradio UI ****. They moved the document processing and RAG pipeline into a function called `process_input` which takes URLs and a question as inputs ****.

```python
import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

def process_input(urls, question):
    # Split the incoming string of URLs into a list
    url_list = urls.split("\n")
    
    # Define the model
    model = ChatOllama(model="mistral")
    
    # Retrieve and split data
    loader = WebBaseLoader(url_list)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # Convert to embeddings and store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="ui_collection"
    )
    
    retriever = vector_db.as_retriever()
    
    # Define the After RAG template and chain
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
    
    # Invoke the chain with the user's question
    return after_rag_chain.invoke(question)

# Setup Gradio Interface
interface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(lines=3, label="Enter URLs (one per line)"),
        gr.Textbox(lines=1, label="Ask a Question")
    ],
    outputs="text",
    title="Ollama RAG Application"
)

# Launch the app
interface.launch()
```
To run the user interface, you would type `python ui.py` in your terminal ****. Once loaded, you can paste URLs into the first box, ask a question like "what is olama" in the second box, and click submit to receive an answer based strictly on the content of the provided websites ****.
