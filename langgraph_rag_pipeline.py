from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from pydantic import BaseModel
from collections import defaultdict
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
import os
import operator

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    question: str
    standalone_question: str
    query_variations: List[str]
    retrieved_docs: List[List[Document]]
    fused_docs: List[Document]
    reranked_docs: List[Document]
    answer: str


class QueryVariations(BaseModel):
    queries: List[str]

#embedding_model = OpenAIEmbeddings(
#    model="text-embedding-3-small"
#)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
#    model_kwargs={"device": "cpu"},   # use "cuda" if GPU available
#    encode_kwargs={"normalize_embeddings": True}
)

#model = ChatOpenAI(model="gpt-4o")

#llm = HuggingFaceEndpoint(
#    repo_id="Qwen/Qwen3-8B",
#    task="text-generation",
#    max_new_tokens=512,
#    temperature=0.3
#)

#model = ChatHuggingFace(llm=llm)

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

persistent_directory = "db/chroma_db"

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("APPLYING RECIPROCAL RANK FUSION")

    rrf_scores = defaultdict(float)
    all_unique_chunks = {}

    for chunks in chunk_lists:

        for position, chunk in enumerate(chunks, 1):

            chunk_content = chunk.page_content

            all_unique_chunks[chunk_content] = chunk

            score = 1 / (k + position)

            rrf_scores[chunk_content] += score

    sorted_chunks = sorted(
        [
            (all_unique_chunks[content], score)
            for content, score in rrf_scores.items()
        ],
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_chunks

def rewrite_question(state: AgentState):

    question = state["question"]
    messages = state.get("messages", [])

    if messages:

        prompt = [
            SystemMessage(
                content="""
                Rewrite the latest user question as a standalone
                searchable question using chat history.
                Return only the rewritten question.
                """
            )
        ] + messages + [
            HumanMessage(content=question)
        ]

        response = model.invoke(prompt)
        standalone_question = response.content.strip()

    else:
        standalone_question = question

    print(f"Standalone Question: {standalone_question}")

    return {
        "standalone_question": standalone_question
    }

def generate_variations(state: AgentState):

    llm = model.with_structured_output(QueryVariations)

    prompt = f"""
    Generate 3 different search query variations for:

    {state["standalone_question"]}

    Return only 3 alternative queries.
    """

    response = llm.invoke(prompt)

    print("Generated Variations:")
    for q in response.queries:
        print(q)

    return {
        "query_variations": response.queries
    }

def retrieve_documents(state: AgentState):

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 20
        }
    )

    all_queries = (
        [state["standalone_question"]]
        + state["query_variations"]
    )

    retrieval_results = []

    for query in all_queries:

        docs = retriever.invoke(query)

        print(f"Retrieved {len(docs)} docs for:")
        print(query)

        retrieval_results.append(docs)

    return {
        "retrieved_docs": retrieval_results
    }

def rrf_node(state: AgentState):

    fused_results = reciprocal_rank_fusion(
        state["retrieved_docs"]
    )

    fused_docs = [
        doc for doc, score in fused_results
    ]

    return {
        "fused_docs": fused_docs
    }

def rerank_documents(state: AgentState):

    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=10
    )

    reranked_docs = reranker.compress_documents(
        state["fused_docs"],
        state["question"]
    )

    return {
        "reranked_docs": reranked_docs
    }

def generate_answer(state: AgentState):

    docs_text = "\\n\\n".join(
        doc.page_content
        for doc in state["reranked_docs"]
    )

    prompt = f"""
    Answer the question using ONLY the documents.

    Question:
    {state["question"]}

    Documents:
    {docs_text}

    If answer is unavailable, say:
    I don't have enough information.
    """

    response = model.invoke([
        SystemMessage(
            content="You answer questions using retrieved documents only."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content)
        ]
    }

def build_graph():

    builder = StateGraph(AgentState)

    builder.add_node("rewrite_question", rewrite_question)
    builder.add_node("generate_variations", generate_variations)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("rrf", rrf_node)
    builder.add_node("rerank", rerank_documents)
    builder.add_node("answer", generate_answer)

    builder.set_entry_point("rewrite_question")

    builder.add_edge("rewrite_question", "generate_variations")
    builder.add_edge("generate_variations", "retrieve_documents")
    builder.add_edge("retrieve_documents", "rrf")
    builder.add_edge("rrf", "rerank")
    builder.add_edge("rerank", "answer")
    builder.add_edge("answer", END)

    return builder.compile()

def load_documents(docs_path):

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"{docs_path} does not exist."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader
    )

    docs = loader.load()

    if len(docs) == 0:
        raise FileNotFoundError(
            "No PDF files found."
        )

    return docs

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\\n\\n", "\\n", ".", ",", " ", ""]
    )

    return splitter.split_documents(docs)

def create_vector_database(chunks):

    vectorstore = Chroma.from_documents(
        documents=chunks,
        persist_directory=persistent_directory,
        embedding=embedding_model,
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    return vectorstore

def setup_vectorstore():

    if os.path.exists(persistent_directory):

        print("Loading existing vector database...")

        return Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model
        )

    print("Creating vector database...")

    docs = load_documents("data")
    chunks = split_documents(docs)

    return create_vector_database(chunks)

def start_chat():

    graph = build_graph()

    messages = []

    print("Ask questions. Type 'quit' to exit.")

    while True:

        question = input("Your Question: ")

        if question.lower() == "quit":
            break

        result = graph.invoke({
            "question": question,
            "messages": messages
        })

        print("Answer:")
        print(result["answer"])

        messages.extend(result["messages"])

if __name__ == "__main__":

    db = setup_vectorstore()

    start_chat()
