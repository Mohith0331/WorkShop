from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from collections import defaultdict
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
import os
import operator

load_dotenv()

docs_path = 'data'
PERSIST_DIRECTORY = "db/chroma_db"


def create_vector_store():
    print("Loading documents...")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )
    docs = loader.load()

    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    #    model_kwargs={"device": "cpu"},   # use "cuda" if GPU available
    #    encode_kwargs={"normalize_embeddings": True}
    )


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return vectorstore


def generate_queries(question, llm):
    prompt = f"""
You are an AI assistant.

Generate 4 different search queries that capture
different perspectives of the user's question.

Question: {question}

Return one query per line only.
"""

    response = llm.invoke(prompt)

    queries = [
        q.strip("- ").strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

    queries.insert(0, question)

    return queries[:5]


def retrieve_documents(queries, retriever):
    all_docs = []

    for query in queries:
        docs = retriever.invoke(query)
        all_docs.append(docs)

    return all_docs


def reciprocal_rank_fusion(results, k=60):
    fused_scores = defaultdict(float)
    doc_map = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content

            fused_scores[doc_id] += 1 / (rank + k)
            doc_map[doc_id] = doc

    reranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    fused_docs = [doc_map[doc] for doc, _ in reranked]

    return fused_docs


def fusion_rag(question, retriever, model):
    print("\nGenerating multiple queries...")
    queries = generate_queries(question, llm)

    print("\nGenerated Queries:")
    for i, q in enumerate(queries, start=1):
        print(f"{i}. {q}")

    print("\nRetrieving documents...")
    retrieved_docs = retrieve_documents(queries, retriever)

    print("Applying Reciprocal Rank Fusion...")
    fused_docs = reciprocal_rank_fusion(retrieved_docs)

    context = "\n\n".join(
        [doc.page_content for doc in fused_docs[:5]]
    )

    prompt = f"""
Answer the question only using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.invoke(prompt)

    return response.content


def main():
    vectorstore = create_vector_store()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    #llm = ChatOpenAI(
    #    model="gpt-4.1-mini",
    #    temperature=0
    #)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    while True:
        question = input("\nAsk a question ('exit' to quit): ")

        if question.lower() == "exit":
            break

        answer = fusion_rag(
            question,
            retriever,
            model
        )

        print("\n" + "=" * 80)
        print("ANSWER:\n")
        print(answer)
        print("=" * 80)


if __name__ == "__main__":
    main()
