from pathlib import Path
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def find_all_docs(folder_path):
    pdf_files = list(Path(folder_path).rglob("*.pdf"))
    txt_files = list(Path(folder_path).rglob("*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in '{folder_path}'. "
            "Please check the folder path or ensure there are supported files"
        )
    
    return all_files


def load_and_split_docs(docs_paths, chunk_size=500, chunk_overlap=50):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    print(f"Found {len(docs_paths)} PDF files.")
    for path in docs_paths:
        print(f"   -> processing: {path.name}")

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            print(f"   -> Skipped unsupported file: {path.name}")
            continue

        pages = loader.load()
        docs = splitter.split_documents(pages)
        # add source info to metadata
        for doc in docs:
            doc.metadata['source'] = path.name
        all_docs.extend(docs)
    
    print(f"Total chunks created: {len(all_docs)}")
    return all_docs