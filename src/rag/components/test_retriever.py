import os
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True) 

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "vectorstore", "db_faiss")


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
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {pdf_path}: {e}")
            
    return docs

def load_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(DB_FAISS_PATH):
        print("Loading existing FAISS database from disk...")
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    docs = load_pdfs()
    if not docs:
        print("No documents found to index.")
        return None

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=120,
    ).split_documents(docs)

    print("Creating and saving new FAISS database...")
    db = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print("Database saved successfully!")

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
            source_name = os.path.basename(r.metadata.get('source', 'Unknown File'))
            page_num = r.metadata.get('page', 0) + 1
            
            print(f"[Result {i}] [File: {source_name} - Page: {page_num}]")
            print(f"{r.page_content}\n")
            print("-" * 50)
