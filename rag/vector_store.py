"""
RAG layer: builds a FAISS vector index over uploaded job-description
PDFs and exposes a retriever used by the Career Advisor agent to
ground its answers in real job postings instead of hallucinating
requirements.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_TOP_K, EMBED_MODEL


def build_vector_store(job_description_docs: list[dict]):
    """
    job_description_docs: list of {"filename": str, "text": str}
    Returns a FAISS vector store, or None if no docs were provided.
    """
    if not job_description_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    for doc in job_description_docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"source": doc["filename"], "chunk": i},
                )
            )

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    return vector_store


def get_retriever(vector_store, k: int = RETRIEVER_TOP_K):
    if vector_store is None:
        return None
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_relevant_jd_context(retriever, query: str) -> str:
    """
    Pulls the most relevant job-description chunks for a query
    (e.g. the resume's target role) and formats them with source
    attribution for citation in the final output.
    """
    if retriever is None:
        return ""

    docs = retriever.invoke(query)
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)
