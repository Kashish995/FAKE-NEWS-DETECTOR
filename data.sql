-- Database Schema for Fake News Detection Logger (news_logs.db)
-- Designed for SQLite / Relational Persistence

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    news_text TEXT NOT NULL,
    prediction TEXT NOT NULL,         -- 'Fake' or 'Real'
    confidence REAL NOT NULL,           -- Probability score (0.0 to 1.0)
    feedback TEXT DEFAULT NULL,         -- User verification feedback ('Correct' or 'Incorrect')
    similarity_score REAL DEFAULT NULL, -- Cosine similarity score from live NewsAPI Search
    clickbait_score REAL DEFAULT 0.0,   -- Sensation/clickbait index score (0.0 to 100.0)
    credibility_score REAL DEFAULT 70.0, -- Domain trust scorer rating (0.0 to 100.0)
    username TEXT DEFAULT 'guest'       -- User identifier for per-user history tracking
);

CREATE TABLE IF NOT EXISTS live_corpus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    title TEXT NOT NULL,
    url TEXT
);