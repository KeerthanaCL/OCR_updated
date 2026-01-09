import cv2
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PreprocessingEvaluator:
    """
    Evaluate preprocessing effectiveness by comparing OCR results
    before and after preprocessing.
    """
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compare_preprocessing(
        self,
        original_image_path: str,
        preprocessed_image_path: str,
        ocr_service,
        ground_truth: str = None
    ) -> Dict:
        """
        Compare OCR results with and without preprocessing.
        
        Args:
            original_image_path: Path to original image
            preprocessed_image_path: Path to preprocessed image
            ocr_service: OCR service instance (Tesseract/TrOCR/EasyOCR)
            ground_truth: Known correct text (optional, for accuracy calculation)
            
        Returns:
            Dict with comparison metrics
        """
        logger.info("Starting preprocessing comparison...")
        
        # Extract text from both versions
        try:
            # Original image
            text_original, conf_original, meta_original = ocr_service.extract_text_with_confidence(
                original_image_path,
                preprocess=False
            )
            
            # Preprocessed image
            text_preprocessed, conf_preprocessed, meta_preprocessed = ocr_service.extract_text_with_confidence(
                preprocessed_image_path,
                preprocess=False  # Already preprocessed
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {"error": str(e)}
        
        # Calculate image quality metrics
        original_quality = self._calculate_image_quality(original_image_path)
        preprocessed_quality = self._calculate_image_quality(preprocessed_image_path)
        
        # Build comparison results
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "original": {
                "text": text_original,
                "confidence": conf_original,
                "word_count": len(text_original.split()),
                "char_count": len(text_original),
                "quality_metrics": original_quality,
                "metadata": meta_original
            },
            "preprocessed": {
                "text": text_preprocessed,
                "confidence": conf_preprocessed,
                "word_count": len(text_preprocessed.split()),
                "char_count": len(text_preprocessed),
                "quality_metrics": preprocessed_quality,
                "metadata": meta_preprocessed
            },
            "improvements": {
                "confidence_gain": conf_preprocessed - conf_original,
                "confidence_gain_percent": ((conf_preprocessed - conf_original) / conf_original * 100) if conf_original > 0 else 0,
                "word_count_change": len(text_preprocessed.split()) - len(text_original.split()),
                "quality_improvement": self._compare_quality(original_quality, preprocessed_quality)
            }
        }
        
        # Calculate accuracy if ground truth provided
        if ground_truth:
            results["accuracy"] = {
                "original": self._calculate_accuracy(text_original, ground_truth),
                "preprocessed": self._calculate_accuracy(text_preprocessed, ground_truth),
            }
            results["accuracy"]["improvement"] = (
                results["accuracy"]["preprocessed"] - results["accuracy"]["original"]
            )
        
        # Save results
        self._save_comparison(results, original_image_path)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _calculate_image_quality(self, image_path: str) -> Dict[str, float]:
        """
        Calculate objective image quality metrics.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # 2. Contrast (standard deviation of pixel intensities)
        contrast = img.std()
        
        # 3. Brightness (mean pixel intensity)
        brightness = img.mean()
        
        # 4. Signal-to-Noise Ratio (SNR)
        mean = np.mean(img)
        std = np.std(img)
        snr = mean / std if std > 0 else 0
        
        # 5. Edge density (measure of detail)
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            "sharpness": round(sharpness, 2),
            "contrast": round(contrast, 2),
            "brightness": round(brightness, 2),
            "snr": round(snr, 2),
            "edge_density": round(edge_density, 4)
        }
    
    def _compare_quality(self, original: Dict, preprocessed: Dict) -> Dict:
        """Compare quality metrics"""
        return {
            "sharpness_gain": round(preprocessed["sharpness"] - original["sharpness"], 2),
            "contrast_gain": round(preprocessed["contrast"] - original["contrast"], 2),
            "snr_gain": round(preprocessed["snr"] - original["snr"], 2)
        }
    
    def _calculate_accuracy(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate accuracy using Character Error Rate (CER).
        Lower CER = better accuracy.
        """
        # Normalize texts
        predicted = predicted.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # Calculate Levenshtein distance (edit distance)
        distance = self._levenshtein_distance(predicted, ground_truth)
        
        # Calculate CER
        cer = distance / len(ground_truth) if len(ground_truth) > 0 else 0
        
        # Convert to accuracy percentage
        accuracy = max(0, (1 - cer) * 100)
        
        return round(accuracy, 2)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _save_comparison(self, results: Dict, image_path: str):
        """Save comparison results to JSON"""
        filename = Path(image_path).stem
        output_path = self.output_dir / f"{filename}_comparison.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison saved to: {output_path}")
    
    def _print_summary(self, results: Dict):
        """Print comparison summary to console"""
        print("\n" + "="*70)
        print("📊 PREPROCESSING EVALUATION RESULTS")
        print("="*70)
        
        print("\n🔍 OCR CONFIDENCE:")
        print(f"  Original:      {results['original']['confidence']:.2f}%")
        print(f"  Preprocessed:  {results['preprocessed']['confidence']:.2f}%")
        print(f"  Improvement:   {results['improvements']['confidence_gain']:+.2f}% "
              f"({results['improvements']['confidence_gain_percent']:+.2f}%)")
        
        print("\n📝 TEXT EXTRACTION:")
        print(f"  Original words:      {results['original']['word_count']}")
        print(f"  Preprocessed words:  {results['preprocessed']['word_count']}")
        print(f"  Change:              {results['improvements']['word_count_change']:+d}")
        
        print("\n🖼️  IMAGE QUALITY:")
        orig_q = results['original']['quality_metrics']
        prep_q = results['preprocessed']['quality_metrics']
        print(f"  Sharpness:   {orig_q['sharpness']:.2f} → {prep_q['sharpness']:.2f} "
              f"({results['improvements']['quality_improvement']['sharpness_gain']:+.2f})")
        print(f"  Contrast:    {orig_q['contrast']:.2f} → {prep_q['contrast']:.2f} "
              f"({results['improvements']['quality_improvement']['contrast_gain']:+.2f})")
        print(f"  SNR:         {orig_q['snr']:.2f} → {prep_q['snr']:.2f} "
              f"({results['improvements']['quality_improvement']['snr_gain']:+.2f})")
        
        if "accuracy" in results:
            print("\n✅ ACCURACY (vs Ground Truth):")
            print(f"  Original:      {results['accuracy']['original']:.2f}%")
            print(f"  Preprocessed:  {results['accuracy']['preprocessed']:.2f}%")
            print(f"  Improvement:   {results['accuracy']['improvement']:+.2f}%")
        
        # Overall verdict
        print("\n" + "="*70)
        if results['improvements']['confidence_gain'] > 5:
            print("✅ VERDICT: Preprocessing SIGNIFICANTLY IMPROVED OCR accuracy")
        elif results['improvements']['confidence_gain'] > 0:
            print("✅ VERDICT: Preprocessing IMPROVED OCR accuracy")
        elif results['improvements']['confidence_gain'] > -5:
            print("⚠️  VERDICT: Preprocessing had MINIMAL IMPACT")
        else:
            print("❌ VERDICT: Preprocessing DEGRADED OCR accuracy")
        print("="*70 + "\n")
    
    def batch_evaluate(
        self,
        test_images: List[str],
        ocr_service,
        ground_truths: List[str] = None
    ) -> Dict:
        """
        Evaluate preprocessing on multiple test images.
        
        Args:
            test_images: List of image paths
            ocr_service: OCR service instance
            ground_truths: Optional list of ground truth texts
            
        Returns:
            Aggregated statistics
        """
        from app.services.preprocessing import AdvancedImagePreprocessor
        
        preprocessor = AdvancedImagePreprocessor(enable_super_resolution=True)
        
        results = []
        
        for idx, image_path in enumerate(test_images):
            logger.info(f"\n📄 Evaluating image {idx+1}/{len(test_images)}: {image_path}")
            
            # Preprocess image
            preprocessed = preprocessor.preprocess_for_ocr(image_path)
            preprocessed_path = self.output_dir / f"preprocessed_{Path(image_path).name}"
            preprocessed.save(preprocessed_path)
            
            # Compare
            ground_truth = ground_truths[idx] if ground_truths else None
            result = self.compare_preprocessing(
                image_path,
                str(preprocessed_path),
                ocr_service,
                ground_truth
            )
            
            results.append(result)
        
        # Calculate aggregate statistics
        avg_confidence_gain = np.mean([r['improvements']['confidence_gain'] for r in results])
        avg_sharpness_gain = np.mean([r['improvements']['quality_improvement']['sharpness_gain'] for r in results])
        
        summary = {
            "total_images": len(test_images),
            "average_confidence_gain": round(avg_confidence_gain, 2),
            "average_sharpness_gain": round(avg_sharpness_gain, 2),
            "improved_count": sum(1 for r in results if r['improvements']['confidence_gain'] > 0),
            "degraded_count": sum(1 for r in results if r['improvements']['confidence_gain'] < 0),
            "individual_results": results
        }
        
        # Save summary
        summary_path = self.output_dir / "batch_evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 BATCH EVALUATION SUMMARY:")
        print(f"  Total images:            {summary['total_images']}")
        print(f"  Improved:                {summary['improved_count']}")
        print(f"  Degraded:                {summary['degraded_count']}")
        print(f"  Avg confidence gain:     {summary['average_confidence_gain']:+.2f}%")
        print(f"  Avg sharpness gain:      {summary['average_sharpness_gain']:+.2f}")
        
        return summary