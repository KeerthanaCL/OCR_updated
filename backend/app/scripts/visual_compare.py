import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from backend.app.services.preprocessing import AdvancedImagePreprocessor

def visualize_preprocessing(image_path: str):
    """Create side-by-side visual comparison"""
    
    # Load original
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    preprocessor = AdvancedImagePreprocessor(enable_super_resolution=False)
    preprocessed = preprocessor.preprocess_for_ocr(image_path)
    preprocessed_np = np.array(preprocessed)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(preprocessed_np, cmap='gray')
    axes[1].set_title("Preprocessed Image", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("./preprocessing_comparison.png", dpi=150)
    plt.show()
    
    print("Visual comparison saved to: preprocessing_comparison.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    else:
        print("Usage: python visual_compare.py <image_path>")
        print("Example: python visual_compare.py ./uploads/test.png")
        sys.exit(1)
    
    visualize_preprocessing(IMAGE_PATH)