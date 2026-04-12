import logging
from pathlib import Path

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(__file__).with_name("AIAYN.pdf")
PERSIST_DIRECTORY = Path(__file__).with_name("chroma-db")
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"
QUERY = "What is the core mechanism of the Transformer architecture?"


def build_vectorstore(pdf_path: Path, embeddings: OllamaEmbeddings) -> Chroma:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Loading docs...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks")

    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
    )


def main() -> None:
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = build_vectorstore(PDF_PATH, embeddings)

    print(f"\nTesting retrieval with query: '{QUERY}'")

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 15},
    )
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    print("Expanding queries and searching database...\n")
    results = advanced_retriever.invoke(QUERY)

    if not results:
        print("No results returned.")
        return

    print("--- Top Retrieved Chunks ---")
    for i, res in enumerate(results, start=1):
        page = res.metadata.get("page", "Unknown")
        preview = res.page_content[:400].replace("\n", " ")
        print(f"\n[Chunk {i}] (Page {page})")
        print(f"{preview}...\n")


if __name__ == "__main__":
    main() 