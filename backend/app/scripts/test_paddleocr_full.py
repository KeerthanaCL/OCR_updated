"""
Full PaddleOCR test with PaddleOCR 3.x API
"""
from paddleocr import PaddleOCR
import time

print("=" * 70)
print("🚀 PADDLEOCR FULL TEST (v3.x)")
print("=" * 70)

# Initialize PaddleOCR with new API
print("\n🔧 Initializing PaddleOCR...")
start = time.time()

# PaddleOCR 3.x simplified API
ocr = PaddleOCR(lang="en")

print(f"✅ PaddleOCR initialized in {time.time() - start:.2f}s")

print("\n📄 PaddleOCR is ready to use!")
print("Usage: result = ocr.ocr('path/to/image.png')")
print("=" * 70)