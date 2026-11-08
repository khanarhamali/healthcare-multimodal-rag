import os
import glob
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..","data")
INDEX_NAME = "skincare-rag-index"


def load_pdfs():
    if not os.path.exists(DATA_DIR):
        print(f"Directory Not Found: {DATA_DIR}")
        return []
    
    docs = []
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return []

    print(f"Found {len(pdf_files)} PDFs to process...")
    
    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_path, extract_tables="markdown")
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata["file_name"] = os.path.basename(pdf_path)
                
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading file {pdf_path}: {e}")
            
    return docs


def load_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()

    if stats.total_vector_count > 0:
        print(f"Connecting to existing Pinecone index ({stats.total_vector_count} vectors found)...")
        return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    docs = load_pdfs()
    if not docs:
        print("No documents found to index.")
        return None

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=120,
    ).split_documents(docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    print("Uploading chunks with metadata to Pinecone...")
    db = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("Data and metadata uploaded to Pinecone successfully!")

    return db


def build_retriever():
    store = load_store()
    if store is None:
        raise ValueError("Vector store failed to initialize.")
    return store.as_retriever(search_kwargs={"k": 5})


if __name__ == "__main__":
    retriever = build_retriever()
    query = "what is the care for oily skin"
    
    print(f"\n--- Invoking Retriever with query: '{query}' ---")
    results = retriever.invoke(query)
    
    if not results:
        print("No Match Found In Database")
    else:
        print(f"Total {len(results)} relevant chunks mile:\n")
        for i, r in enumerate(results, 1):
            source_name = r.metadata.get('file_name', os.path.basename(r.metadata.get('source', 'Unknown File')))
            page_num = int(r.metadata.get('page', 0)) + 1
            chunk_idx = int(r.metadata.get('chunk_index', 0))
            
            print(f"[Result {i}] [File: {source_name} | Page: {page_num} | Chunk: {chunk_idx}]")
            print(f"{r.page_content}\n")
            print("-" * 50)
