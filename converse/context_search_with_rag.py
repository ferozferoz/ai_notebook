import time
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Configuration & Connection Properties
# Format: postgresql+psycopg://username:password@host:port/database_name
CONNECTION_STRING = "postgresql+psycopg://postgres:yourpassword@localhost:5432/your_db"
COLLECTION_NAME = "local_knowledge_base"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:1.5b"

print("🔄 Initializing Ollama models...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

# 2. Setup Postgres Vector Store via LangChain
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

def ingest_sample_data():
    """Simulates loading data, chunking text, and saving to Postgres."""
    print("\n📥 Simulating file ingestion...")

    # Mock data representing proprietary configuration settings
    raw_documents = [
        Document(
            page_content="The payment gateway module uses an internal secret retry timeout config parameter named 'GATEWAY_RETRY_MS' set to exactly 4500 milliseconds. Do not set it lower than 4000 to prevent race conditions.",
            metadata={"source": "billing_specs.txt"}
        ),
        Document(
            page_content="Database maintenance happens automatically every Sunday at 02:00 AM UTC. Connection pools are dropped during this 5 minute window.",
            metadata={"source": "ops_runbook.md"}
        )
    ]

    # Text splitting breaks pages into optimal overlapping windows
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    docs = text_splitter.split_documents(raw_documents)

    print(f"💾 Saving {len(docs)} text chunks to your PostgreSQL vector tables...")
    vector_store.add_documents(docs)
    print("✅ Ingestion successfully persistent inside Postgres.")

def query_rag_system(user_question: str):
    """Searches Postgres vectors and pipes matches to Ollama code generation."""
    print(f"\n🔍 Processing Question: '{user_question}'")

    # 1. Similarity Search: Converts question to vector and asks Postgres for nearest matches
    print("📡 Fetching matching context fragments from Postgres...")
    relevant_chunks = vector_store.similarity_search(user_question, k=2)

    # 2. Extract and compile matching fragments
    context_text = "\n---\n".join([chunk.page_content for chunk in relevant_chunks])
    print(f"📌 Context found (from {len(relevant_chunks)} snippets)")

    # 3. Formulate the Augmented Prompt Blueprint
    system_prompt = (
        "You are an assistant answering technical questions strictly based on the following local context records.\n"
        "If the context does not contain the answer, tell the user you do not know.\n\n"
        f"CONTEXT RECORDS:\n{context_text}\n\n"
        f"USER QUESTION: {user_question}\n"
        "ANSWER:"
    )

    # 4. Stream response generation from local LLM
    print("🤖 Ollama Generating Answer:\n")
    stream = llm.stream(system_prompt)
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    # Note: Run ingestion once to populate the DB.
    # You can comment out ingest_sample_data() on subsequent runs!
    ingest_sample_data()

    time.sleep(1) # Small rest window for db syncing

    # Test queries showing vector math matching context
    query_rag_system("What is the exact timeout limit name for the gateway and what is it set to?")
    query_rag_system("When does database maintenance take place?")
