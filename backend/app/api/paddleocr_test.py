"""
PaddleOCR Testing API
Standalone endpoint to test PaddleOCR performance vs Tesseract
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
import logging

from app.database import get_db, Document
from app.services.paddleocr_service import PaddleOCRService
from app.services.tesseract_service import TesseractService
from app.services.storage import StorageService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/test", tags=["paddle-testing"])

# Initialize services (lazy loading)
paddle_service = None
tesseract_service = None


class PaddleOCRRequest(BaseModel):
    """Request for PaddleOCR testing"""
    document_id: str
    use_preprocessing: bool = False


class ComparisonRequest(BaseModel):
    """Request for comparing Tesseract vs PaddleOCR"""
    document_id: str
    use_preprocessing: bool = False


@router.post("/paddleocr")
async def test_paddleocr(
    request: PaddleOCRRequest,
    db: Session = Depends(get_db)
):
    """
    Test PaddleOCR extraction only.
    
    Returns:
    - Extracted text
    - Confidence score
    - Processing time
    - GPU usage stats
    """
    global paddle_service
    
    try:
        # Get document
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get file path
        storage = StorageService()
        file_path = storage.get_file_path(request.document_id)
        
        logger.info(f"Testing PaddleOCR")
        
        # Initialize PaddleOCR service (lazy)
        if paddle_service is None:
            paddle_service = PaddleOCRService()
        
        # Extract text
        start_time = time.time()
        
        if request.use_preprocessing:
            text, confidence, metadata = paddle_service.extract_text_region_based(
                file_path, 
                preprocess=True
            )
        else:
            text, confidence, metadata = paddle_service.extract_text(file_path)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"PaddleOCR completed: confidence={confidence:.2f}%, "
            f"words={len(text.split())}, time={processing_time:.2f}s"
        )
        
        return {
            "success": True,
            "engine": "paddleocr",
            "document_id": request.document_id,
            "filename": document.filename,
            "text": text,
            "confidence": confidence,
            "word_count": len(text.split()),
            "char_count": len(text),
            "processing_time": processing_time,
            "gpu_enabled": metadata.get("gpu_enabled", False),
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"PaddleOCR test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_engines(
    request: ComparisonRequest,
    db: Session = Depends(get_db)
):
    """
    Compare Tesseract (CPU) vs PaddleOCR (GPU) side-by-side.
    
    Returns:
    - Results from both engines
    - Performance comparison
    - Winner recommendation
    """
    global paddle_service, tesseract_service
    
    try:
        # Get document
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get file path
        storage = StorageService()
        file_path = storage.get_file_path(request.document_id)
        
        logger.info(f"Starting comparison: Tesseract vs PaddleOCR")
        
        # Initialize services (lazy)
        if tesseract_service is None:
            tesseract_service = TesseractService()
        
        if paddle_service is None:
            paddle_service = PaddleOCRService(use_gpu=True, gpu_id=request.gpu_id)
        
        # Test Tesseract
        logger.info("Testing Tesseract (CPU)...")
        start_tess = time.time()
        text_tess, conf_tess, meta_tess = tesseract_service.extract_text_region_based(
            file_path,
            preprocess=request.use_preprocessing
        )
        time_tess = time.time() - start_tess
        
        # Test PaddleOCR
        logger.info("Testing PaddleOCR (GPU)...")
        start_paddle = time.time()
        
        if request.use_preprocessing:
            text_paddle, conf_paddle, meta_paddle = paddle_service.extract_text_region_based(
                file_path,
                preprocess=True
            )
        else:
            text_paddle, conf_paddle, meta_paddle = paddle_service.extract_text(file_path)
        
        time_paddle = time.time() - start_paddle
        
        # Calculate comparison metrics
        speedup = time_tess / time_paddle if time_paddle > 0 else 0
        conf_improvement = conf_paddle - conf_tess
        
        # Determine winner
        winner = "paddleocr" if conf_paddle > conf_tess else "tesseract"
        winner_reason = []
        
        if conf_paddle > conf_tess + 5:
            winner_reason.append(f"Higher confidence (+{conf_improvement:.2f}%)")
        if time_paddle < time_tess:
            winner_reason.append(f"Faster ({speedup:.2f}x speedup)")
        if len(text_paddle.split()) > len(text_tess.split()):
            word_diff = len(text_paddle.split()) - len(text_tess.split())
            winner_reason.append(f"More words extracted (+{word_diff})")
        
        logger.info(
            f"Comparison complete: Winner = {winner.upper()} "
            f"(Tesseract: {conf_tess:.2f}%, {time_tess:.2f}s | "
            f"PaddleOCR: {conf_paddle:.2f}%, {time_paddle:.2f}s)"
        )
        
        return {
            "success": True,
            "document_id": request.document_id,
            "filename": document.filename,
            
            # Tesseract Results
            "tesseract": {
                "text": text_tess,
                "confidence": conf_tess,
                "word_count": len(text_tess.split()),
                "char_count": len(text_tess),
                "processing_time": time_tess,
                "method": meta_tess.get("method", "region_based"),
                "metadata": meta_tess
            },
            
            # PaddleOCR Results
            "paddleocr": {
                "text": text_paddle,
                "confidence": conf_paddle,
                "word_count": len(text_paddle.split()),
                "char_count": len(text_paddle),
                "processing_time": time_paddle,
                "gpu_enabled": meta_paddle.get("gpu_enabled", False),
                "metadata": meta_paddle
            },
            
            # Comparison
            "comparison": {
                "confidence_improvement": conf_improvement,
                "confidence_improvement_percent": (conf_improvement / conf_tess * 100) if conf_tess > 0 else 0,
                "speedup": speedup,
                "time_saved": time_tess - time_paddle,
                "word_count_difference": len(text_paddle.split()) - len(text_tess.split()),
                "winner": winner,
                "winner_reasons": winner_reason,
                "recommendation": (
                    f"Use {winner.upper()}: " + ", ".join(winner_reason)
                    if winner_reason
                    else f"Both engines perform similarly"
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gpu-info")
async def get_gpu_info():
    """
    Get GPU information for PaddleOCR.
    Shows available GPUs and their status.
    """
    try:
        import paddle
        
        gpu_available = paddle.device.is_compiled_with_cuda()
        
        if not gpu_available:
            return {
                "gpu_available": False,
                "message": "CUDA not available. PaddlePaddle was not compiled with CUDA support.",
                "recommendation": "Install paddlepaddle-gpu instead of paddlepaddle"
            }
        
        gpu_count = paddle.device.cuda.device_count()
        current_device = paddle.device.get_device()
        
        gpus = []
        for i in range(gpu_count):
            try:
                paddle.device.set_device(f'gpu:{i}')
                gpu_name = paddle.device.cuda.get_device_name(i)
                memory_info = paddle.device.cuda.mem_get_info(i)
                free_memory = memory_info[0] / (1024**3)  # Convert to GB
                total_memory = memory_info[1] / (1024**3)
                
                gpus.append({
                    "gpu_id": i,
                    "name": gpu_name,
                    "free_memory_gb": round(free_memory, 2),
                    "total_memory_gb": round(total_memory, 2),
                    "used_memory_gb": round(total_memory - free_memory, 2),
                    "usage_percent": round((1 - free_memory/total_memory) * 100, 2)
                })
            except Exception as e:
                logger.warning(f"Failed to get info for GPU {i}: {e}")
        
        return {
            "gpu_available": True,
            "gpu_count": gpu_count,
            "current_device": current_device,
            "gpus": gpus,
            "recommendation": (
                f"Use GPU {gpus[0]['gpu_id']} ({gpus[0]['name']}) "
                f"with {gpus[0]['free_memory_gb']:.1f}GB free memory"
                if gpus else "No GPU info available"
            )
        }
        
    except ImportError:
        return {
            "gpu_available": False,
            "error": "PaddlePaddle not installed",
            "recommendation": "Install with: pip install paddlepaddle-gpu"
        }
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-compare")
async def batch_compare(
    document_ids: list[str],
    gpu_id: int = 1,
    db: Session = Depends(get_db)
):
    """
    Compare Tesseract vs PaddleOCR on multiple documents.
    Returns aggregate statistics.
    """
    global paddle_service, tesseract_service
    
    if len(document_ids) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 documents allowed per batch"
        )
    
    try:
        # Initialize services
        if tesseract_service is None:
            tesseract_service = TesseractService()
        if paddle_service is None:
            paddle_service = PaddleOCRService(use_gpu=True, gpu_id=gpu_id)
        
        storage = StorageService()
        results = []
        
        for doc_id in document_ids:
            document = db.query(Document).filter(Document.id == doc_id).first()
            if not document:
                logger.warning(f"Document {doc_id} not found, skipping")
                continue
            
            file_path = storage.get_file_path(doc_id)
            
            # Tesseract
            start = time.time()
            text_tess, conf_tess, _ = tesseract_service.extract_text_region_based(file_path)
            time_tess = time.time() - start
            
            # PaddleOCR
            start = time.time()
            text_paddle, conf_paddle, _ = paddle_service.extract_text(file_path)
            time_paddle = time.time() - start
            
            results.append({
                "document_id": doc_id,
                "filename": document.filename,
                "tesseract_confidence": conf_tess,
                "paddleocr_confidence": conf_paddle,
                "confidence_improvement": conf_paddle - conf_tess,
                "tesseract_time": time_tess,
                "paddleocr_time": time_paddle,
                "speedup": time_tess / time_paddle if time_paddle > 0 else 0,
                "winner": "paddleocr" if conf_paddle > conf_tess else "tesseract"
            })
        
        # Calculate aggregate stats
        avg_conf_improvement = sum(r["confidence_improvement"] for r in results) / len(results)
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        paddle_wins = sum(1 for r in results if r["winner"] == "paddleocr")
        
        return {
            "success": True,
            "total_documents": len(results),
            "results": results,
            "summary": {
                "average_confidence_improvement": round(avg_conf_improvement, 2),
                "average_speedup": round(avg_speedup, 2),
                "paddleocr_wins": paddle_wins,
                "tesseract_wins": len(results) - paddle_wins,
                "recommendation": (
                    f"Use PaddleOCR: {paddle_wins}/{len(results)} wins, "
                    f"{avg_conf_improvement:+.2f}% avg confidence, {avg_speedup:.2f}x faster"
                    if paddle_wins > len(results) / 2
                    else f"PaddleOCR advantage unclear, more testing needed"
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Batch comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))