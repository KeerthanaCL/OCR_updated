from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import get_settings

settings = get_settings()

# Create engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    """Document database model"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)
    status = Column(String, default="uploaded")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Extraction(Base):
    """Extraction results database model"""
    __tablename__ = "extractions"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, index=True)
    text = Column(Text)
    confidence = Column(Float)
    method_used = Column(String)
    pages = Column(Integer)
    processing_time = Column(Float)
    extraction_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Parsing(Base):
    """Parsing results database model"""
    __tablename__ = "parsings"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, index=True)
    extraction_id = Column(String)
    fields = Column(JSON)
    parsing_method = Column(String)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()