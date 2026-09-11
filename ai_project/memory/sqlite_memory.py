import sqlite3
from datetime import datetime
import os 

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "memory.db")
)
print("🧠 SQLite DB PATH =", DB_PATH)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS  chat_memory(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        timestamp TEXT
        
        )
        """
         )
         # 2️⃣ 🧠 新增：长期记忆表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory_summary (
        session_id TEXT PRIMARY KEY,
        summary TEXT,
        updated_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_message(session_id:str,role:str,content:str):
    conn = sqlite3.connect(DB_PATH)
    print("💾 SAVE MESSAGE:", session_id, role, content)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO  chat_memory(session_id,role,content,timestamp)
        VALUES(?,?,?,?)

        """,(session_id,role,content,datetime.now().isoformat())

    )
    conn.commit()
    conn.close()

def save_summary(session_id: str, summary: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print("💾 SAVE SUMMARY:", session_id, summary)
    cursor.execute("""
    INSERT INTO memory_summary (session_id, summary, updated_at)
    VALUES (?, ?, ?)
    ON CONFLICT(session_id)
    DO UPDATE SET
        summary=excluded.summary,
        updated_at=excluded.updated_at
    """, (session_id, summary, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def get_summary(session_id: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT summary FROM memory_summary
    WHERE session_id=?
    """, (session_id,))

    row = cursor.fetchone()
    conn.close()

    return row[0] if row else ""



def get_history(session_id,limit:int=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
       """
        SELECT role,content FROM chat_memory
        WHERE session_id=?
        ORDER BY id DESC
        LIMIT ?

       """,(session_id,limit)
    )
    
    rows = cursor.fetchall()
    print("📖 HISTORY RAW:", rows)
    return list(reversed(rows))



    

           