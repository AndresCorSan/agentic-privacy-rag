import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "chat_history.db"

def init_db():
    """Crea las tablas necesarias si no existen."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabla de conversaciones
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP
        )
    ''')
    
    # Tabla de mensajes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT, -- Guardado como JSON string
            steps TEXT,   -- Guardado como JSON string
            timestamp TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def save_conversation(conv_id, title):
    """Guarda o actualiza una conversación."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversations (id, title, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET title=excluded.title
    ''', (conv_id, title, datetime.now()))
    conn.commit()
    conn.close()

def save_message(conv_id, role, content, sources=None, steps=None):
    """Guarda un nuevo mensaje en la base de datos."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    sources_json = json.dumps(sources or [])
    steps_json = json.dumps(steps or [])
    
    cursor.execute('''
        INSERT INTO messages (conversation_id, role, content, sources, steps, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (conv_id, role, content, sources_json, steps_json, datetime.now()))
    
    conn.commit()
    conn.close()

def load_all_conversations():
    """Carga todas las conversaciones y sus mensajes de la base de datos."""
    if not os.path.exists(DB_PATH):
        return {}
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Para acceder por nombre de columna
    cursor = conn.cursor()
    
    # Cargar conversaciones
    cursor.execute('SELECT * FROM conversations ORDER BY created_at ASC')
    convs_rows = cursor.fetchall()
    
    conversations = {}
    for row in convs_rows:
        cid = row['id']
        conversations[cid] = {
            "title": row['title'],
            "messages": []
        }
        
        # Cargar mensajes de esta conversación
        cursor.execute('SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC', (cid,))
        msg_rows = cursor.fetchall()
        for m_row in msg_rows:
            conversations[cid]["messages"].append({
                "role": m_row['role'],
                "content": m_row['content'],
                "sources": json.loads(m_row['sources']),
                "steps": json.loads(m_row['steps']),
                "timestamp": datetime.fromisoformat(m_row['timestamp']) if isinstance(m_row['timestamp'], str) else m_row['timestamp']
            })
            
    conn.close()
    return conversations

def delete_conversation_db(conv_id):
    """Elimina una conversación y sus mensajes de la DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM conversations WHERE id = ?', (conv_id,))
    # SQLite con ON DELETE CASCADE activado borraría los mensajes automáticamente, 
    # pero por seguridad lo hacemos explícito si no está configurado el PRAGMA.
    cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conv_id,))
    conn.commit()
    conn.close()
