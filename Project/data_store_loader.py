# import weaviate
# import faiss
import time
# from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_weaviate.vectorstores import WeaviateVectorStore
import hashlib, json
from pathlib import Path

def file_check_sum(path):
    """
    Return a hash for the file
    """
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_manifest(manifest_path):
    if Path(manifest_path).exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {}

def save_manifest(manifest, manifest_path):
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

def update_vectorstore_from_folder(folder_path, vectorstore, embedding_model, manifest_path="manifest.json"):
    manifest = load_manifest(manifest_path)
    current_files = list(Path(folder_path).rglob("*.pdf")) + list(Path(folder_path).rglob("*.txt"))
    current_state = {str(p): file_check_sum(p) for p in current_files}

    added = []
    removed = []

    start_time = time.time()
    # Detect new or modified files
    for path_str, checksum in current_state.items():
        if path_str not in manifest or manifest[path_str] != checksum:
            added.append(path_str)
    
    # Detect removed files
    for path_str in manifest.keys():
        if path_str not in current_state:
            removed.append(path_str)
    
    # Remove docs
    if removed:
        print(f"Removing {len(removed)} documents from vectorstore...")
        vectorstore.delete(filter={"source": {"$in": removed}})
        for r in removed:
            del manifest[r]

    # Add new/modified docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    for path_str in added:
        print(f"Adding/updating: {path_str}")
        if path_str.endswith(".pdf"):
            loader = PyPDFLoader(path_str)
        elif path_str.endswith(".txt"):
            loader = TextLoader(path_str, autodetect_encoding=True)
        else:
            continue
        try:
            pages = loader.load()
        except Exception as e:
            print(f"load files {path_str} appears error: {e}")
            import traceback
            traceback.print_exc()
            continue
        docs = splitter.split_documents(pages)
        for doc in docs:
            doc.metadata['source'] = path_str
        vectorstore.add_documents(docs)
        manifest[path_str] = current_state[path_str]

    save_manifest(manifest, manifest_path)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Vectorstore updated with {total_time}")

# def embed_and_store(docs, embedding_model, db_type="chroma", persist_dir="./vector_db", **kwargs):
#     """
#     Embed and store documents in a selected vector database

#     Args:
#         docs (list): List of documents to embed.
#         embedding_model; Embedding model instance.
#         db_type (str): Type of vector DB to use. Options: 'chroma', 'faiss', 'weaviate'
#         persist_dir (str): Directory to persist vector DB (if supported).
#         **kwargs: Addtional arguments passed to the vector store constructor

#     Returns:
#         vectorstore: A vector store instance.
#     """

#     print(f"Storing all documents in {db_type} vector DB...")
#     start_time = time.time()
#     if db_type == "chroma":
#         vectorstore = Chroma.from_documents(
#             documents=docs,
#             embedding=embedding_model,
#             persist_directory=persist_dir,
#             **kwargs
#         )
#         vectorstore.persist()

#     # elif db_type == "faiss":
#     #     vectorstore = FAISS.from_documents(
#     #         documents=docs,
#     #         embedding=embedding_model,
#     #         **kwargs
#     #     )
#     #     vectorstore.save_local(persist_dir)
    
#     # elif db_type == "weaviate":
#     #     client = kwargs.get("client")
#     #     if client is None:
#     #         raise ValueError("Weaviate requires a 'client' in kwargs.")
#     #     vectorstore = Weaviate.from_documents(
#     #         documents=docs,
#     #         embedding=embedding_model,
#     #         client=client,
#     #         **kwargs
#     #     )

#     else:
#         raise ValueError(f"Unsupported db_type: {db_type}")
#     end_time = time.time()

#     total_time = end_time - start_time

#     print(f"Vector DB stored using {db_type} at {persist_dir}")
#     print(f"processing time: {total_time}")
#     return vectorstore

# def query_vectorstore(vectorstore, query, k=3):
#     print(f"\n Query: \"{query}\"")
#     results = vectorstore.similarity_search(query, k=k)
#     for i, doc in enumerate(results):
#         print(f"\nResult {i+1}:\n{doc.page_content})")