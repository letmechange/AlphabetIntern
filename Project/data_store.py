# import weaviate
# import faiss
from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import FAISS
# from langchain_weaviate.vectorstores import WeaviateVectorStore


def embed_and_store(docs, embedding_model, db_type="chroma", persist_dir="./vector_db", **kwargs):
    """
    Embed and store documents in a selected vector database

    Args:
        docs (list): List of documents to embed.
        embedding_model; Embedding model instance.
        db_type (str): Type of vector DB to use. Options: 'chroma', 'faiss', 'weaviate'
        persist_dir (str): Directory to persist vector DB (if supported).
        **kwargs: Addtional arguments passed to the vector store constructor

    Returns:
        vectorstore: A vector store instance.
    """

    print(f"Storing all documents in {db_type} vector DB...")

    if db_type == "chroma":
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_dir,
            **kwargs
        )
        vectorstore.persist()

    # elif db_type == "faiss":
    #     vectorstore = FAISS.from_documents(
    #         documents=docs,
    #         embedding=embedding_model,
    #         **kwargs
    #     )
    #     vectorstore.save_local(persist_dir)
    
    # elif db_type == "weaviate":
    #     client = kwargs.get("client")
    #     if client is None:
    #         raise ValueError("Weaviate requires a 'client' in kwargs.")
    #     vectorstore = Weaviate.from_documents(
    #         documents=docs,
    #         embedding=embedding_model,
    #         client=client,
    #         **kwargs
    #     )

    else:
        raise ValueError(f"Unsupported db_type: {db_type}")
    
    print(f"Vector DB stored using {db_type} at {persist_dir}")
    return vectorstore

def query_vectorstore(vectorstore, query, k=3):
    print(f"\n Query: \"{query}\"")
    results = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:\n{doc.page_content})")