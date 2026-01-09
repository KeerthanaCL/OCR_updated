import sqlite3

print("🔧 Adding 'extracted_texts' table to documents.db...\n")

conn = sqlite3.connect("documents.db")
cursor = conn.cursor()

try:
    # Create extracted_texts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extracted_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            confidence REAL,
            method_used TEXT,
            word_count INTEGER,
            char_count INTEGER,
            pages INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES raia_runs(run_id)
        )
    """)
    print("✅ Table 'extracted_texts' created!")

    # Create indexes for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_extracted_texts_run_id 
        ON extracted_texts(run_id)
    """)
    print("✅ Index on 'run_id' created!")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_extracted_texts_document_id 
        ON extracted_texts(document_id)
    """)
    print("✅ Index on 'document_id' created!")

    conn.commit()
    print("\n🎉 Successfully added 'extracted_texts' table and indexes!")
    
    # Verify
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='extracted_texts'")
    if cursor.fetchone():
        print("✅ Verification: Table exists in database")
        
        # Show structure
        cursor.execute("PRAGMA table_info(extracted_texts)")
        columns = cursor.fetchall()
        print("\n📊 Table structure:")
        for col in columns:
            print(f"   {col[1]:<20} {col[2]:<15}")
    else:
        print("❌ Verification failed!")

except Exception as e:
    print(f"❌ Error: {e}")
finally:
    conn.close()