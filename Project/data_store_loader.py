# # import weaviate
# # import faiss
# import time
# # from langchain.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores import FAISS
# # from langchain_weaviate.vectorstores import WeaviateVectorStore
# import hashlib, json
# from pathlib import Path

# def file_check_sum(path):
#     """
#     Return a hash for the file
#     """
#     with open(path, 'rb') as f:
#         return hashlib.md5(f.read()).hexdigest()

# def load_manifest(manifest_path):
#     if Path(manifest_path).exists():
#         with open(manifest_path, 'r') as f:
#             return json.load(f)
#     return {}

# def save_manifest(manifest, manifest_path):
#     with open(manifest_path, 'w') as f:
#         json.dump(manifest, f, indent=2)

# def update_vectorstore_from_folder(folder_path, vectorstore, manifest_path="manifest.json"):
#     manifest = load_manifest(manifest_path)
#     current_files = list(Path(folder_path).rglob("*.pdf")) + list(Path(folder_path).rglob("*.txt"))
#     current_state = {str(p): file_check_sum(p) for p in current_files}

#     added = []
#     removed = []

#     start_time = time.time()
#     # Detect new or modified files
#     for path_str, checksum in current_state.items():
#         if path_str not in manifest or manifest[path_str] != checksum:
#             added.append(path_str)
    
#     # Detect removed files
#     for path_str in manifest.keys():
#         if path_str not in current_state:
#             removed.append(path_str)
    
#     # Remove docs
#     if removed:
#         print(f"Removing {len(removed)} documents from vectorstore...")
#         vectorstore.delete(where={"source": {"$in": removed}})
#         for r in removed:
#             del manifest[r]

#     # Add new/modified docs
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
#     for path_str in added:
#         print(f"Adding/updating: {path_str}")
#         if path_str.endswith(".pdf"):
#             loader = PyPDFLoader(path_str)
#         elif path_str.endswith(".txt"):
#             loader = TextLoader(path_str, autodetect_encoding=True)
#         else:
#             continue
#         try:
#             pages = loader.load()
#         except Exception as e:
#             print(f"load files {path_str} appears error: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
#         docs = splitter.split_documents(pages)
#         for doc in docs:
#             doc.metadata['source'] = path_str
#         vectorstore.add_documents(docs)
#         manifest[path_str] = current_state[path_str]

#     save_manifest(manifest, manifest_path)

#     end_time = time.time()
#     total_time = end_time - start_time

#     print(f"Vectorstore updated with {total_time}")

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


import time, hashlib, json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def file_check_sum(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_manifest(manifest_path):
    if Path(manifest_path).exists():
        with open(manifest_path, 'r', encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(manifest, manifest_path):
    with open(manifest_path, 'w', encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

def add_in_batches(vectorstore, docs, batch_size=2000):
    for i in range(0, len(docs), batch_size):
        vectorstore.add_documents(docs[i:i+batch_size])

def update_vectorstore_from_folder(folder_path, vectorstore, manifest_path="manifest.json"):
    manifest = load_manifest(manifest_path)
    current_files = list(Path(folder_path).rglob("*.pdf")) + list(Path(folder_path).rglob("*.txt"))
    current_state = {str(p): file_check_sum(p) for p in current_files}

    added, removed = [], []
    start_time = time.time()

    for path_str, checksum in current_state.items():
        if path_str not in manifest or manifest[path_str] != checksum:
            added.append(path_str)

    for path_str in list(manifest.keys()):
        if path_str not in current_state:
            removed.append(path_str)

    if removed:
        print(f"Removing {len(removed)} documents from vectorstore...")
        vectorstore.delete(where={"source": {"$in": removed}})
        for r in removed:
            del manifest[r]

    # Chinese/English-friendly splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n","。","！","？","；","：","\n","，","、"," ", ".", "!", "?", ";", ":"],
    )

    total = 0
    for path_str in added:
        print(f"Adding/updating: {path_str}")
        if path_str.endswith(".pdf"):
            loader = PyPDFLoader(path_str)
        elif path_str.endswith(".txt"):
            loader = TextLoader(path_str, autodetect_encoding=True)
        else:
            continue

        # 1) Load + filter + pdfminer fallback
        try:
            pages = loader.load()
        except Exception as e:
            print(f"load files {path_str} appears error: {e}")
            import traceback; traceback.print_exc()
            pages = []

        pages = [p for p in pages if p.page_content and p.page_content.strip()]
        if not pages and path_str.endswith(".pdf"):
            try:
                from pdfminer.high_level import extract_text
                txt = (extract_text(path_str) or "").replace("\u200b","").replace("\ufeff","").strip()
                if txt:
                    from langchain.docstore.document import Document
                    pages = [Document(page_content=txt, metadata={"source": path_str})]
            except Exception:
                pass

        if not pages:
            print(f"[SKIP] No text extracted: {path_str}")
            continue

        # 2) Split + filter
        docs = splitter.split_documents(pages)
        from pathlib import Path as _P
        for d in docs:
            d.metadata['source'] = _P(path_str).resolve().as_posix()

        docs = [d for d in docs if d.page_content and d.page_content.strip()]
        if not docs:
            print(f"[SKIP] Empty after splitting: {path_str}")
            continue

        # 3) Add (batched) + persist + manifest update
        try:
            add_in_batches(vectorstore, docs, batch_size=2000)
            try:
                vectorstore.persist()
            except Exception:
                pass
            manifest[path_str] = current_state[path_str]
            total += len(docs)
            print(f"[OK] {path_str} -> {len(docs)} chunks")
        except Exception as e:
            print(f"[ADD-ERR] {path_str}: {e}")

    save_manifest(manifest, manifest_path)
    print(f"Vectorstore updated with {time.time() - start_time:.2f}s, chunks added: {total}")


# import time, hashlib, json
# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # ---- NEW: deps for CSV/XLSX ----
# import pandas as pd  # pip install pandas
# # For .xlsx, ensure: pip install openpyxl

# # ---- NEW: types & sanitizers for safe metadata ----
# from datetime import date, datetime, time as dtime
# import numpy as np

# def _sanitize_meta_value(v):
#     """Convert metadata values to primitives supported by vector DBs.
#     Allowed: str, int, float, bool, None. Everything else -> str (ISO for dates).
#     """
#     if v is None:
#         return None
#     # pandas NA/NaT
#     if pd.isna(v):
#         return None
#     # pandas Timestamp/Timedelta
#     if isinstance(v, pd.Timestamp):
#         try:
#             return v.isoformat()
#         except Exception:
#             return str(v)
#     if isinstance(v, pd.Timedelta):
#         return str(v)
#     # datetime/date/time
#     if isinstance(v, (datetime, date, dtime)):
#         try:
#             return v.isoformat()
#         except Exception:
#             return str(v)
#     # numpy scalars
#     if isinstance(v, (np.integer,)):
#         return int(v)
#     if isinstance(v, (np.floating,)):
#         fv = float(v)
#         if np.isnan(fv) or np.isinf(fv):
#             return None
#         return fv
#     if isinstance(v, (np.bool_,)):
#         return bool(v)
#     # simple python types
#     if isinstance(v, (str, int, float, bool)):
#         return v
#     # lists/dicts/others => string (compact)
#     return str(v)

# def _sanitize_meta_dict(d: dict) -> dict:
#     return {str(k): _sanitize_meta_value(v) for k, v in (d or {}).items() if _sanitize_meta_value(v) is not None}

# # ========== ORIGINAL HELPERS ==========

# def file_check_sum(path):
#     with open(path, 'rb') as f:
#         return hashlib.md5(f.read()).hexdigest()

# def load_manifest(manifest_path):
#     if Path(manifest_path).exists():
#         with open(manifest_path, 'r', encoding="utf-8") as f:
#             return json.load(f)
#     return {}

# def save_manifest(manifest, manifest_path):
#     with open(manifest_path, 'w', encoding="utf-8") as f:
#         json.dump(manifest, f, indent=2, ensure_ascii=False)

# def add_in_batches(vectorstore, docs, batch_size=2000):
#     for i in range(0, len(docs), batch_size):
#         vectorstore.add_documents(docs[i:i+batch_size])

# # ========== CSV/XLSX → Documents ==========
# # What to embed vs. keep as metadata
# _TEXT_COL_HINTS = {"title","name","summary","description","notes","comment","remarks"}  # case-insensitive
# _MAX_CELL_CHARS = 2000  # guardrail

# def _choose_text_columns(df: pd.DataFrame):
#     text_cols, meta_cols = [], []
#     for col in df.columns:
#         col_lower = str(col).lower()
#         dtype = str(df[col].dtype)
#         if ("object" in dtype or "string" in dtype) or any(k in col_lower for k in _TEXT_COL_HINTS):
#             text_cols.append(col)
#         else:
#             meta_cols.append(col)
#     return text_cols, meta_cols

# def _row_to_page(row: pd.Series, text_cols, meta_cols, file_path: str, sheet_name: str|None, row_idx: int):
#     # Build readable text with headers preserved
#     parts = []
#     for c in text_cols:
#         v = row.get(c, "")
#         if pd.isna(v):
#             continue
#         s = str(v)
#         if len(s) > _MAX_CELL_CHARS:
#             s = s[:_MAX_CELL_CHARS] + "…"
#         parts.append(f"{c}: {s}")
#     page_content = "\n".join(parts).strip()

#     if not page_content:
#         preview = ", ".join(f"{c}={row.get(c)}" for c in text_cols[:6])
#         page_content = f"Row {row_idx} (no long text). Preview: {preview}"

#     # LangChain Document (lazy import to avoid top-level dependency)
#     from langchain_core.documents import Document
#     md = {
#         "source_path": file_path,
#         "sheet": sheet_name,
#         "row_index": int(row_idx),
#     }
#     for c in meta_cols:
#         v = row.get(c, None)
#         sv = _sanitize_meta_value(v)
#         if sv is not None:
#             md[str(c)] = sv

#     md = _sanitize_meta_dict(md)

#     # Stable id per file/sheet/row
#     stable_id = hashlib.sha256(f"{file_path}|{sheet_name}|{row_idx}".encode("utf-8")).hexdigest()[:32]
#     return Document(page_content=page_content, metadata=md, id=stable_id)

# def _dataframe_to_documents(df: pd.DataFrame, file_path: str, sheet_name: str|None):
#     df = df.copy()
#     df.columns = [str(c) for c in df.columns]
#     text_cols, meta_cols = _choose_text_columns(df)
#     docs = []
#     for i, row in df.iterrows():
#         docs.append(_row_to_page(row, text_cols, meta_cols, file_path, sheet_name, i))
#     return docs

# def _tabular_file_to_docs(path: str):
#     """Returns a list of Documents (one per row, with headers preserved in text)."""
#     p = path.lower()
#     if p.endswith(".csv"):
#         df = pd.read_csv(path)
#         return _dataframe_to_documents(df, file_path=path, sheet_name=None)
#     elif p.endswith(".xlsx") or p.endswith(".xls"):
#         docs = []
#         xls = pd.ExcelFile(path)
#         for sheet in xls.sheet_names:
#             df = pd.read_excel(path, sheet_name=sheet)
#             docs.extend(_dataframe_to_documents(df, file_path=path, sheet_name=sheet))
#         return docs
#     else:
#         return []

# # ========== MAIN UPDATE PIPELINE (same shape, extended formats + metadata fix) ==========

# def update_vectorstore_from_folder(folder_path, vectorstore, manifest_path="manifest.json"):
#     manifest = load_manifest(manifest_path)

#     # include csv/xlsx
#     current_files = (
#         list(Path(folder_path).rglob("*.pdf")) +
#         list(Path(folder_path).rglob("*.txt")) +
#         list(Path(folder_path).rglob("*.csv")) +
#         list(Path(folder_path).rglob("*.xlsx")) +
#         list(Path(folder_path).rglob("*.xls"))
#     )
#     current_state = {str(p): file_check_sum(p) for p in current_files}

#     added, removed = [], []
#     start_time = time.time()

#     for path_str, checksum in current_state.items():
#         if path_str not in manifest or manifest[path_str] != checksum:
#             added.append(path_str)

#     for path_str in list(manifest.keys()):
#         if path_str not in current_state:
#             removed.append(path_str)

#     if removed:
#         print(f"Removing {len(removed)} documents from vectorstore...")
#         vectorstore.delete(where={"source": {"$in": removed}})
#         for r in removed:
#             del manifest[r]

#     # Chinese/English-friendly splitter (keep your original)
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=700,
#         chunk_overlap=100,
#         separators=["\n\n","。","！","？","；","：","\n","，","、"," ", ".", "!", "?", ";", ":"],
#     )

#     total = 0
#     for path_str in added:
#         print(f"Adding/updating: {path_str}")
#         docs = []

#         try:
#             lower = path_str.lower()
#             if lower.endswith(".pdf"):
#                 # ---- PDFs ----
#                 loader = PyPDFLoader(path_str)
#                 pages = []
#                 try:
#                     pages = loader.load()
#                 except Exception as e:
#                     print(f"load files {path_str} appears error: {e}")
#                     import traceback; traceback.print_exc()
#                     pages = []

#                 pages = [p for p in pages if p.page_content and p.page_content.strip()]
#                 if not pages:
#                     # pdfminer fallback
#                     try:
#                         from pdfminer_high_level import extract_text  # alias if available
#                     except Exception:
#                         from pdfminer.high_level import extract_text
#                     try:
#                         txt = (extract_text(path_str) or "").replace("\u200b",""").replace("\ufeff",""").strip()
#                         if txt:
#                             from langchain.docstore.document import Document as _D
#                             pages = [_D(page_content=txt, metadata={"source": path_str})]
#                     except Exception:
#                         pass

#                 if not pages:
#                     print(f"[SKIP] No text extracted: {path_str}")
#                     continue

#                 docs = splitter.split_documents(pages)

#             elif lower.endswith(".txt"):
#                 # ---- TXT ----
#                 loader = TextLoader(path_str, autodetect_encoding=True)
#                 try:
#                     pages = loader.load()
#                 except Exception as e:
#                     print(f"load files {path_str} appears error: {e}")
#                     import traceback; traceback.print_exc()
#                     pages = []

#                 pages = [p for p in pages if p.page_content and p.page_content.strip()]
#                 if not pages:
#                     print(f"[SKIP] No text extracted: {path_str}")
#                     continue

#                 docs = splitter.split_documents(pages)

#             elif lower.endswith((".csv", ".xlsx", ".xls")):
#                 # ---- CSV/XLSX ----
#                 try:
#                     base_docs = _tabular_file_to_docs(path_str)  # one Document per row with headers in text
#                 except Exception as e:
#                     print(f"load files {path_str} appears error: {e}")
#                     import traceback; traceback.print_exc()
#                     base_docs = []

#                 if not base_docs:
#                     print(f"[SKIP] No rows extracted: {path_str}")
#                     continue

#                 # Chunk rows as needed (wide/verbose rows)
#                 docs = splitter.split_documents(base_docs)

#             else:
#                 continue

#             # Normalize metadata to match your deletion filter and sanitize
#             from pathlib import Path as _P
#             for d in docs:
#                 d.metadata = _sanitize_meta_dict({**(d.metadata or {}), "source": _P(path_str).resolve().as_posix()})

#             docs = [d for d in docs if d.page_content and d.page_content.strip()]
#             if not docs:
#                 print(f"[SKIP] Empty after splitting: {path_str}")
#                 continue

#             # 3) Add (batched) + persist + manifest update
#             try:
#                 add_in_batches(vectorstore, docs, batch_size=2000)
#                 try:
#                     vectorstore.persist()
#                 except Exception:
#                     pass
#                 manifest[path_str] = current_state[path_str]  # only after success
#                 total += len(docs)
#                 print(f"[OK] {path_str} -> {len(docs)} chunks")
#             except Exception as e:
#                 print(f"[ADD-ERR] {path_str}: {e}")

#         except Exception as e:
#             print(f"[ERR] {path_str}: {e}")

#     save_manifest(manifest, manifest_path)
#     print(f"Vectorstore updated with {time.time() - start_time:.2f}s, chunks added: {total}")

