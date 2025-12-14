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

class AIDetectionResult(Base):
    """AI Detection results storage"""
    __tablename__ = "ai_detection_results"
    
    id = Column(String, primary_key=True, index=True)
    extraction_id = Column(String, index=True, nullable=False)
    is_ai_generated = Column(Boolean)
    confidence = Column(Float)
    detection_method = Column(String)
    flags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    raw_response = Column(JSON)

class HorizonResult(Base):
    """Horizon segmentation results"""
    __tablename__ = "horizon_results"
    
    id = Column(String, primary_key=True, index=True)
    extraction_id = Column(String, index=True, nullable=False)
    segment_type = Column(String)  # 'references', 'medical', 'legal'
    data = Column(JSON)
    validation = Column(JSON)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)

class OrchestrationJob(Base):
    """Job tracking for orchestrated processing"""
    __tablename__ = "orchestration_jobs"
    
    id = Column(String, primary_key=True, index=True)
    extraction_id = Column(String, index=True, nullable=False)
    status = Column(String, default="processing")  # processing, complete, failed
    progress = Column(Integer, default=0)  # 0-100
    
    # Results (stored as JSON, null until complete)
    ai_detection_result = Column(JSON)
    references_result = Column(JSON)
    medical_result = Column(JSON)
    legal_result = Column(JSON)
    
    # Tracking
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()