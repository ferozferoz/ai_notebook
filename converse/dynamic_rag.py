import os
import time
import hashlib
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
CONNECTION_STRING = "postgresql+psycopg://postgres:yourpassword@localhost:5432/your_db"
COLLECTION_NAME = "dynamic_realtime_knowledge"
WATCH_DIRECTORY = "./live_context"

# Initialize Models
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen2.5-coder:1.5b", temperature=0.1)

# Establish Persistent Connection Engine
engine = PGEngine.from_connection_string(connection_string=CONNECTION_STRING)
vector_store = PGVectorStore(embedding=embeddings, collection_name=COLLECTION_NAME, engine=engine)

# Ensure the local folder exists
os.makedirs(WATCH_DIRECTORY, exist_ok=True)

# --- INGESTION WORKER ENGINE ---
class LiveDirectoryWatcher(FileSystemEventHandler):
    """Listens to OS file system signals and updates Postgres incrementally."""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            print(f"\n⚡ Detected change in: {event.src_path}. Processing update...")
            self.process_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            print(f"\n⚡ New file detected: {event.src_path}. Processing ingestion...")
            self.process_file(event.src_path)

    def process_file(self, file_path):
        # Prevent crash if file is temporarily locked by the OS while writing
        time.sleep(0.5)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return

            # Clean up old records from this specific file first to avoid duplication
            # (This maps perfectly to a Change Data Capture cleanup pattern)
            self.delete_old_file_vectors(file_path)

            # Generate chunks
            filename = os.path.basename(file_path)
            raw_doc = Document(page_content=content, metadata={"source": filename, "full_path": file_path})
            chunks = self.splitter.split_documents([raw_doc])

            # Inject explicit deterministic IDs based on the text hash.
            # This allows Postgres pgvector to run efficient UPSERT lookups.
            ids = [hashlib.md5(f"{filename}_{i}_{c.page_content}".encode()).hexdigest() for i, c in enumerate(chunks)]

            vector_store.add_documents(chunks, ids=ids)
            print(f"✅ Successfully synchronized {len(chunks)} updated chunks to Postgres.")

        except Exception as e:
            print(f"❌ Error processing file sync: {e}")

    def delete_old_file_vectors(self, file_path):
        """Purges existing vectors linked to this filename before writing new ones."""
        filename = os.path.basename(file_path)
        with engine.connect() as conn:
            # Under the hood, LangChain maps metadata to a JSONB column named 'cstore' or 'custom_metadata'
            # We execute a direct SQL sweep targeting chunks matching this source file
            query = "DELETE FROM langchain_pg_embedding WHERE cstore->>'source' = %s"
            conn.execute(query, (filename,))
            conn.commit()

# Helper function to spin up the file monitor system in a separate CPU thread
def start_watcher_thread():
    event_handler = LiveDirectoryWatcher()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIRECTORY, recursive=False)
    observer.start()
    print(f"📡 Dynamic RAG Ingestion Worker Active. Watching folder: '{WATCH_DIRECTORY}'")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# --- QUERY ENGINE ---
def ask_realtime_rag(question: str):
    """Queries the database executing a real-time vector scan on the freshest state."""
    print(f"\n🙋 User Question: {question}")

    # Instantly fetches the nearest rows currently sitting in the live Postgres tables
    matching_nodes = vector_store.similarity_search(question, k=2)

    context = "\n---\n".join([node.page_content for node in matching_nodes])

    prompt = (
        "You are an operations dashboard assistant. Answer the user question based strictly "
        "on the following live stream records. If you do not know, say so.\n\n"
        f"LIVE STREAMS:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )

    print("🤖 Response: ", end="")
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
    print("\n")

# --- EXECUTION SIMULATION ---
if __name__ == "__main__":
    # 1. Start the live-updating pipeline background thread
    watcher_thread = Thread(target=start_watcher_thread, daemon=True)
    watcher_thread.start()
    time.sleep(1) # Allow thread to bind securely

    print("\n--- SIMULATION RUNNING ---")
    print("Step A: Creating an initial log file...")

    # Simulate writing an initial text file into the watched folder
    initial_log = os.path.join(WATCH_DIRECTORY, "system_status.txt")
    with open(initial_log, "w", encoding="utf-8") as f:
        f.write("System Status: All services operating normally. Database cluster replication latency is 12ms.")

    time.sleep(2) # Give the background ingestion thread a moment to compute embeddings

    # Query 1
    ask_realtime_rag("What is the current replication latency of the database?")

    print("Step B: Simulating a live production event update...")
    # Overwrite the exact same file with completely new real-time log data
    with open(initial_log, "w", encoding="utf-8") as f:
        f.write("System Status: CRITICAL ALERT. Database cluster replication latency has spiked to 9450ms due to network throttling.")

    time.sleep(2) # Wait for automated vector update pipeline execution

    # Query 2 (The model should seamlessly report the brand new state)
    ask_realtime_rag("What is the current replication latency of the database?")

    # Keep script alive so you can test manually editing files in the folder!
    print("💡 You can now manually open the 'live_context' folder and change the text files. The RAG pipeline will adapt in real time.")
    while True:
        user_q = input("\nEnter a manual test question (or type 'exit' to quit): ")
        if user_q.lower() == 'exit':
            break
        ask_realtime_rag(user_q)
