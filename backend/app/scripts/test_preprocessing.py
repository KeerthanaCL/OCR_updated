import sys
sys.path.append('./backend')

from app.services.tesseract_service import TesseractService
from app.utils.preprocessing_evaluator import PreprocessingEvaluator

def test_single_image(image_path=None):
    """Test preprocessing on a single image"""
    # Initialize services
    ocr_service = TesseractService()
    evaluator = PreprocessingEvaluator(output_dir="./test_results")
    
    # Use provided path or default
    if image_path is None:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            print("Usage: python test_preprocessing.py <image_path>")
            print("Example: python test_preprocessing.py ./uploads/test.png")
            return
    
    # Optional: ground truth
    ground_truth = """..."""
    
    # Run comparison
    results = evaluator.compare_preprocessing(
        original_image_path=image_path,
        preprocessed_image_path=image_path,
        ocr_service=ocr_service,
        ground_truth=ground_truth
    )

if __name__ == "__main__":
    test_single_image()