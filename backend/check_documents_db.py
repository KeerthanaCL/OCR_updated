import sqlite3
from pathlib import Path

# Database path
db_path = Path("documents.db")

if not db_path.exists():
    print(f"❌ Database not found at: {db_path.absolute()}")
    exit(1)

print(f"✅ Found database: {db_path.absolute()}\n")

# Connect to database
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# List all tables
print("📋 Tables in documents.db:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = cursor.fetchall()
for table in tables:
    print(f"   - {table[0]}")

print("\n" + "="*80 + "\n")

# Check if extracted_texts table exists
cursor.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name='extracted_texts'
""")
table_exists = cursor.fetchone()

if not table_exists:
    print("❌ Table 'extracted_texts' DOES NOT EXIST in documents.db")
    print("\n⚠️  The table was not created. This means _create_tables() didn't run properly.")
    print("\nPossible reasons:")
    print("   1. The fix to __init__ method wasn't applied yet")
    print("   2. Server needs to be restarted")
    print("   3. Database was created before the fix")
else:
    print("✅ Table 'extracted_texts' EXISTS!\n")
    
    # Show table structure
    cursor.execute("PRAGMA table_info(extracted_texts)")
    columns = cursor.fetchall()
    print("📊 Table structure:")
    for col in columns:
        print(f"   - {col[1]:<20} {col[2]:<10}")
    
    # Count records
    cursor.execute("SELECT COUNT(*) FROM extracted_texts")
    count = cursor.fetchone()[0]
    print(f"\n📈 Total records: {count}")
    
    if count > 0:
        print("\n📝 Recent extracted texts:\n")
        cursor.execute("""
            SELECT 
                run_id,
                document_id,
                word_count,
                confidence,
                method_used,
                pages,
                created_at,
                substr(extracted_text, 1, 100) as text_preview
            FROM extracted_texts
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        for row in cursor.fetchall():
            print(f"Run ID: {row[0][:30]}...")
            print(f"Document: {row[1]}")
            print(f"Words: {row[2]} | Confidence: {row[3]:.1f}% | Method: {row[4]} | Pages: {row[5]}")
            print(f"Created: {row[6]}")
            print(f"Preview: {row[7]}...")
            print("-" * 80)
    else:
        print("\n⚠️  No extracted texts saved yet.")
        print("   Process a document to see data appear here.")

conn.close()