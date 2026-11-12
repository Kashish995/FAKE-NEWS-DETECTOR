import sqlite3
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from .config import DB_FILE

def get_connection():
    return sqlite3.connect(str(DB_FILE))

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            input_mode TEXT DEFAULT 'Text',
            feedback TEXT DEFAULT NULL,
            similarity_score REAL DEFAULT NULL,
            clickbait_score REAL DEFAULT 0.0,
            credibility_score REAL DEFAULT 70.0,
            username TEXT DEFAULT 'guest',
            language TEXT DEFAULT 'English'
        )
    """)
    # Database schema migration checks
    c.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in c.fetchall()]
    if "input_mode" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN input_mode TEXT DEFAULT 'Text'")
    if "clickbait_score" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN clickbait_score REAL DEFAULT 0.0")
    if "credibility_score" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN credibility_score REAL DEFAULT 70.0")
    if "username" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN username TEXT DEFAULT 'guest'")
    if "language" not in columns:
        c.execute("ALTER TABLE predictions ADD COLUMN language TEXT DEFAULT 'English'")

    c.execute("""
        CREATE TABLE IF NOT EXISTS live_corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            title TEXT NOT NULL,
            url TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(
    news_text: str,
    prediction: str,
    confidence: float,
    input_mode: str = "Text",
    similarity_score: Optional[float] = None,
    clickbait_score: float = 0.0,
    credibility_score: float = 70.0,
    username: str = "guest",
    language: str = "English"
) -> Optional[int]:
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO predictions (
                news_text, prediction, confidence, input_mode,
                similarity_score, clickbait_score, credibility_score,
                username, language
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            news_text, prediction, confidence, input_mode,
            similarity_score, clickbait_score, credibility_score,
            username, language
        ))
        row_id = c.lastrowid
        conn.commit()
        conn.close()
        return row_id
    except Exception as e:
        print(f"Database Insert Error: {e}")
        return None

def save_feedback(log_id: int, feedback_value: str) -> bool:
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            UPDATE predictions
            SET feedback = ?
            WHERE id = ?
        """, (feedback_value, log_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database Feedback Error: {e}")
        return False

def get_history(
    target_user: Optional[str] = None,
    limit: int = 100,
    search_query: Optional[str] = None
) -> pd.DataFrame:
    conn = get_connection()
    query = """
        SELECT id, timestamp, username, input_mode, language,
               SUBSTR(news_text, 1, 90) || '...' AS text_preview,
               prediction, ROUND(confidence * 100, 1) AS confidence_pct,
               clickbait_score, credibility_score, feedback
        FROM predictions
    """
    params = []
    conditions = []
    
    if target_user:
        conditions.append("username = ?")
        params.append(target_user)
        
    if search_query and search_query.strip():
        conditions.append("(news_text LIKE ? OR prediction LIKE ?)")
        params.append(f"%{search_query.strip()}%")
        params.append(f"%{search_query.strip()}%")
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += f" ORDER BY id DESC LIMIT {int(limit)}"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_analytics_summary(target_user: Optional[str] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    conn = get_connection()
    c = conn.cursor()
    user_filter = "WHERE username = ?" if target_user else ""
    params = (target_user,) if target_user else ()
    
    c.execute(f"SELECT COUNT(*) FROM predictions {user_filter}", params)
    total = c.fetchone()[0] or 0
    
    # 3-Tier Counts (including backwards compatibility with legacy 'Fake' / 'Real' labels)
    c.execute(f"""
        SELECT 
            SUM(CASE WHEN prediction IN ('Likely Reliable', 'Real') THEN 1 ELSE 0 END),
            SUM(CASE WHEN prediction IN ('Likely Misleading', 'Fake') THEN 1 ELSE 0 END),
            SUM(CASE WHEN prediction = 'Needs Further Verification' THEN 1 ELSE 0 END),
            SUM(CASE WHEN feedback = 'Correct' THEN 1 ELSE 0 END),
            SUM(CASE WHEN feedback = 'Incorrect' THEN 1 ELSE 0 END)
        FROM predictions {user_filter}
    """, params)
    row = c.fetchone()
    
    reliable_count = row[0] or 0
    misleading_count = row[1] or 0
    verify_count = row[2] or 0
    correct_fb = row[3] or 0
    incorrect_fb = row[4] or 0
    
    df_logs = get_history(target_user=target_user, limit=50)
    conn.close()
    
    return {
        "total": total,
        "reliable": reliable_count,
        "misleading": misleading_count,
        "needs_verification": verify_count,
        "correct_feedback": correct_fb,
        "incorrect_feedback": incorrect_fb
    }, df_logs

def clear_logs(target_user: Optional[str] = None) -> bool:
    try:
        conn = get_connection()
        c = conn.cursor()
        if target_user:
            c.execute("DELETE FROM predictions WHERE username = ?", (target_user,))
        else:
            c.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error clearing logs: {e}")
        return False

def get_prediction_by_id(log_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single prediction record by ID for session restoration."""
    try:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT id, timestamp, news_text, prediction, confidence, input_mode,
                   clickbait_score, credibility_score, username, language, feedback
            FROM predictions
            WHERE id = ?
        """, (log_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Error fetching record {log_id}: {e}")
        return None
