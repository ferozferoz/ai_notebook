from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
def ingest_file_folder(folder_path):
    folder_path = Path(folder_path)
    if folder_path.is_file():
        ingest_file(folder_path)
    elif folder_path.is_dir():
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                ingest_file(file_path)


def ingest_file(path):
    loaded_documents = []
    if path.suffix == '.txt':
        loader = TextLoader(str(path), encoding='utf-8')
    elif path.suffix == '.pdf':
        loader = PyPDFLoader(str(path))
    elif path.suffix == '.docx':
        loader = Docx2txtLoader(str(path))
    else:
        print(f"Unsupported file type: {path.suffix}. Skipping {path.name}.")
        return
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    loaded_documents.extend(loader.load())
    final_chunks = text_splitter.split_documents(loaded_documents)
    print(f"Vectorizing and saving {len(final_chunks)} dense research chunks to Postgres...")
    vector_store.add_documents(final_chunks)

def run_research(query):
    print(f"running query {query} ... ")
    node = vector_store.similarity_search(query, k=6)
    context_fragments = [doc.page_content for doc in node]
    compiled_context = "\n".join(context_fragments)

    # create a research prompt with the compiled context and the user query
    research_prompt = (
        "You are a research assistant. Use the following context to answer the question below. "
        "If the context does not contain the answer, respond with 'I don't know'.\n\n"
        f"CONTEXT:\n{compiled_context}\n\n"
        f"QUESTION: {query}\n"
        "ANSWER:"
    )
    print("\n✍️ Generating Structural Synthesis:\n")
    for chunk in llm.stream(research_prompt):
        print(chunk.content, end="", flush=True)
    print("\n")



if __name__ == "__main__":

    folder_input = input("Enter the folder path to analyse (or type 'exit' to quit): ")
    if folder_input.lower() in ['exit', 'quit'] or not folder_input:
        print("Exiting the program.")
        exit(1)

    file_or_folder = folder_input.split("\\")[-1]  # Get the last part of the path
    if file_or_folder:
        COLLECTION_NAME = file_or_folder
    else:
        COLLECTION_NAME = "default_collection"

    CONNECTION_STRING = "postgresql+psycopg://admin:SuperSecurePassword123!@localhost:5433/app_dev"
    EMBED_MODEL = "nomic-embed-text"   # this model is used for vectorizing text
    LLM_MODEL = "qwen2.5-coder:1.5b"   # this model is used for generating answers

    print("Initializing Ollama models...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1,top_p=0.9)
    print("Initializing vector store...")
    # 2. Setup Postgres Vector Store via LangChain
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    print("ingesting sample data...")
    # Note: Run ingestion once to populate the DB.
    # You can comment out ingest_sample_data() on subsequent runs!
    ingest_file_folder(folder_input)
    while True:
        user_question = input("\nEnter your question (or type 'exit' to quit): ")
        if user_question.lower() in ['exit', 'quit'] or not user_question:
            print("Exiting the program.")
            break
        run_research(user_question)