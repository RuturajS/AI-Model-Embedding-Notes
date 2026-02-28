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
