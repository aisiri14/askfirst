"""
database.py — SQLAlchemy models + session management for AskFirst chat app.
Stores threads and messages. Temporary chats bypass this entirely.
"""

import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, ForeignKey, Boolean, func
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from contextlib import contextmanager

DATABASE_URL = "sqlite:///./askfirst.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ──────────────────────────────────────────────────────────────────

class Thread(Base):
    __tablename__ = "threads"

    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String(200), nullable=False, default="New Chat")
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.datetime.utcnow,
                         onupdate=datetime.datetime.utcnow)
    is_temporary = Column(Boolean, default=False)   # kept for reference; temp threads never persisted

    messages = relationship(
        "Message",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def to_dict(self):
        return {
            "id":         self.id,
            "title":      self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages),
        }


class Message(Base):
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, index=True)
    thread_id  = Column(Integer, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String(20), nullable=False)   # "user" | "assistant" | "system"
    content    = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=True)   # which LLM answered
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    thread = relationship("Thread", back_populates="messages")

    def to_dict(self):
        return {
            "id":         self.id,
            "thread_id":  self.thread_id,
            "role":       self.role,
            "content":    self.content,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat(),
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables on startup."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session() -> Session:
    """Context-manager style session (used by background tasks / non-FastAPI code)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def create_thread(db: Session, title: str = "New Chat") -> Thread:
    thread = Thread(title=title)
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


def get_all_threads(db: Session) -> list[Thread]:
    return (
        db.query(Thread)
        .order_by(Thread.updated_at.desc())
        .all()
    )


def get_thread(db: Session, thread_id: int) -> Thread | None:
    return db.query(Thread).filter(Thread.id == thread_id).first()


def delete_thread(db: Session, thread_id: int) -> bool:
    thread = get_thread(db, thread_id)
    if not thread:
        return False
    db.delete(thread)
    db.commit()
    return True


def rename_thread(db: Session, thread_id: int, new_title: str) -> Thread | None:
    thread = get_thread(db, thread_id)
    if not thread:
        return None
    thread.title = new_title[:200]
    thread.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(thread)
    return thread


def add_message(
    db: Session,
    thread_id: int,
    role: str,
    content: str,
    model_used: str | None = None,
) -> Message:
    msg = Message(
        thread_id=thread_id,
        role=role,
        content=content,
        model_used=model_used,
    )
    db.add(msg)
    # bump thread's updated_at
    thread = get_thread(db, thread_id)
    if thread:
        thread.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(msg)
    return msg


def get_thread_messages(db: Session, thread_id: int) -> list[Message]:
    return (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at)
        .all()
    )


def get_global_memory_summary(db: Session, exclude_thread_id: int | None = None, limit: int = 40) -> list[dict]:
    """
    Fetch the most recent `limit` messages across ALL threads (excluding the
    active one) to inject as universal memory context into the system prompt.
    """
    q = db.query(Message).join(Thread)
    if exclude_thread_id:
        q = q.filter(Message.thread_id != exclude_thread_id)
    messages = q.order_by(Message.created_at.desc()).limit(limit).all()
    # Return in chronological order
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]
