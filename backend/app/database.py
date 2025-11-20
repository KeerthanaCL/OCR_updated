from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Text, Boolean
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
    created_at = Column(DateTime, default=datetime.utcnow)
    extraction_metadata = Column(JSON)

class Parsing(Base):
    """Parsing results database model"""
    __tablename__ = "parsings"
    
    id = Column(String, primary_key=True, index=True)
    extraction_id = Column(String, index=True)
    parsed_data = Column(JSON)
    parsing_method = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    parsing_metadata = Column(JSON)

class AppealsExtraction(Base):
    """Appeals extraction tracking"""
    __tablename__ = "appeals_extractions"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, index=True)
    extraction_id = Column(String)  # Link to Extraction table
    appeals_text = Column(Text)
    appeals_found = Column(Boolean, default=False)
    total_confidence = Column(Float)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    appeals_metadata = Column(JSON)

class AppealsSegment(Base):
    """Individual segments from appeals"""
    __tablename__ = "appeals_segments"
    
    id = Column(String, primary_key=True, index=True)
    appeals_extraction_id = Column(String, index=True)
    segment_type = Column(String)  # references, medical_context, legal_context
    content = Column(Text)
    start_position = Column(Integer)
    end_position = Column(Integer)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class SegmentValidation(Base):
    """OpenAI validation results"""
    __tablename__ = "segment_validations"
    
    id = Column(String, primary_key=True, index=True)
    segment_id = Column(String, index=True)
    status = Column(String)  # valid, invalid, partially_valid, uncertain
    confidence_score = Column(Float)
    reasoning = Column(Text)
    issues_found = Column(JSON)
    suggestions = Column(JSON)
    openai_model_used = Column(String)
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