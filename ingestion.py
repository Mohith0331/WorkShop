from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

#embedding_model = AzureOpenAIEmbeddings(
#    model="text-embedding-3-small",
#    azure_endpoint=AZURE_ENDPOINT,
#    api_key=AZURE_API_KEY,
#    api_version="2024-02-01"
#)

#embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
#    model_kwargs={"device": "cpu"},   # use "cuda" if GPU available
#    encode_kwargs={"normalize_embeddings": True}
)

#embedding_model = GoogleGenerativeAIEmbeddings(
#    model="models/gemini-embedding-001"
#)

#persistent_directory = "db/chroma_db"
#persistent_directory = "gem/chroma_db"
#persistent_directory = "gpt/chroma_db"
persistent_directory = "hf/chroma_db"

def load_documents(docs_path):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    print(f"Loading the documents")
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )
    docs = loader.load()

    #loader = DirectoryLoader(
    #    path=docs_path,
    #    glob="*.pdf",
    #    loader_cls=UnstructuredPDFLoader,
    #    loader_kwargs={
    #        "mode": "elements",
    #        "strategy": "fast"
    #    }
    #)

    docs = loader.load()
    
    if len(docs)==0:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}. Please add your company documents.")
    
    print("Documents are loaded")
    
    return docs

def split_documents(docs):
    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n","\n",".",","," ",""]
    )
    chunks = splitter.split_documents(docs)
    #chunks = filter_complex_metadata(chunks)

    return chunks

def create_vector_database(chunks):
    print("Creating embeddings and storing in ChromaDB...")

    vectorStore = Chroma.from_documents(
        documents=chunks,
        persist_directory=persistent_directory,
        embedding=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
    )

    print(f"Vector store created and saved to {persistent_directory}")

    return vectorStore

def main():
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")

        vectorStore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hsnw:space":"cosine"}
        )

        print(f"Loaded existing vector store with vectorstore documents")

        return vectorStore
    
    documents = load_documents("data")

    chunks = split_documents(documents)

    vectorstore = create_vector_database(chunks)

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")

    return vectorstore

if __name__ == "__main__":
    main()