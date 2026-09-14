import psycopg
import urllib

# Paste your connection string here

raw_password = "SuperSecurePassword123!"
safe_password = urllib.parse.quote_plus(raw_password)

# 2. Build the encoded string safely for LangChain
CONNECTION_STRING = f"postgresql+psycopg://admin:{safe_password}@localhost:5433/app_dev"


print("📡 Connecting to PostgreSQL...")

try:
    # Attempt to connect with a strict 3-second timeout so it doesn't hang
    with psycopg.connect(CONNECTION_STRING, connect_timeout=3) as conn:
        print("✅ SUCCESS: Connected to the database container!")

        # Test executing a basic database command
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            db_version = cur.fetchone()
            print(f"🖥️  Postgres Version: {db_version[0]}")

            # Check if pgvector is enabled
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            ext = cur.fetchone()
            if ext:
                print("🟢 pgvector extension: INSTALLED and ready!")
            else:
                print("🔴 pgvector extension: NOT FOUND! Run 'CREATE EXTENSION vector;' in your DB.")

except psycopg.OperationalError as e:
    print("\n❌ CONNECTION FAILED!")
    print(f"Error Details:\n{e}")
