from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank
from pydantic import BaseModel
from collections import defaultdict
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]

# Connect to your document database
#persistent_directory = "db/chroma_db"
#persistent_directory = "gem/chroma_db"
#persistent_directory = "gpt/chroma_db"
persistent_directory = "hf/chroma_db"

#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#embeddings = AzureOpenAIEmbeddings(
#    model="text-embedding-3-small",
#    azure_endpoint=AZURE_ENDPOINT,
#    api_key=AZURE_API_KEY,
#    api_version="2024-02-01"
#)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
#    model_kwargs={"device": "cpu"},   # use "cuda" if GPU available
#    encode_kwargs={"normalize_embeddings": True}
)

#embeddings = GoogleGenerativeAIEmbeddings(
#    model="models/gemini-embedding-001"
#)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up AI model
#model = ChatOpenAI(model="gpt-4o")

#model = AzureChatOpenAI(
#    azure_deployment="gpt-4o",
#    api_version="2024-10-21",
#    temperature=1,
#)


#llm = HuggingFaceEndpoint(
#    repo_id="Qwen/Qwen3-8B",
#    task="text-generation",
#    max_new_tokens=512,
#    temperature=0.3
#)

#model = ChatHuggingFace(llm=llm)


#model = ChatOpenAI(
#    model="deepseek/deepseek-chat-v3-0324",
#    api_key="your_openrouter_api_key",
#    base_url="https://openrouter.ai/api/v1",
#    temperature=0.3
#)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# Store our conversation as messages
chat_history = []

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("APPLYING RECIPROCAL RANK FUSION")
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

def ask_question(user_question):
    print(f"Original Query: {user_question}\n")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    llm_with_tools = model.with_structured_output(QueryVariations)

    prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

    Original query: {search_question}

    Return 3 alternative queries that rephrase or approach the same question from different angles."""

    response = llm_with_tools.invoke(prompt)
    query_variations = response.queries

    #response = model.invoke(prompt)
    #query_variations = response.content.split("\n")

    print("Generated Query Variations:")
    for i, variation in enumerate(query_variations, 1):
        print(f"{i}. {variation}")

    # Step 2: Find relevant documents
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k":20})
    
    all_queries = [search_question] + query_variations
    all_retrieval_results = []
    for i, query in enumerate(all_queries, 1):
        print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
        docs = retriever.invoke(query)
        all_retrieval_results.append(docs)
        print(f"Retrieved {len(docs)} documents:\n")
        for j, doc in enumerate(docs, 1):
            print(f"Document {j}:")
            print(f"{doc.page_content[:150]}...\n")    

    print("Multi-Query Retrieval Complete!")
    print("Notice how different query variations retrieved different documents.")

    #docs = retriever.invoke(search_question)
    
    fused_results = reciprocal_rank_fusion(
        all_retrieval_results,
        k=60,
        verbose=True
    )

    # RRF returns (document, score)
    fused_docs = [
        doc
        for doc, score in fused_results
    ]
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=10
    )
    reranked_docs = reranker.compress_documents(
        fused_docs,
        user_question
    )
    documents_text = "\n\n".join(
        doc.page_content
        for doc in reranked_docs
    )

    combined_input = f"""Based on the following documents, please answer this question: {user_question}
    
    Documents:
    {documents_text}
    
    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."""
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()