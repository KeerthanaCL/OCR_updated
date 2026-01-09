import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class PDFConverter:
    """
    Service for converting PDF documents to images.
    Supports two methods: pdf2image (Poppler) and PyMuPDF (fitz).
    """
    
    def __init__(self, method: str = "pymupdf", save_images: bool = True):
        """
        Initialize PDF converter.
        
        Args:
            method: Conversion method ('pymupdf' or 'pdf2image')
        """
        self.method = method
        self.dpi = 300  # Standard DPI for OCR (300-400 recommended)
        self.save_images = save_images 
        
    def convert_with_pymupdf(
        self, 
        pdf_path: str, 
        output_dir: str
    ) -> List[str]:
        """
        Convert PDF to images using PyMuPDF (fitz).
        
        PyMuPDF is faster and provides better control over quality.
        Recommended for production use.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            
        Returns:
            List of image file paths
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Open PDF
            doc = fitz.open(pdf_path)
            image_paths = []
            
            # Set zoom factor for desired DPI
            # Matrix calculation: dpi / 72 (72 is default PDF DPI)
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            
            logger.info(f"Converting PDF with {len(doc)} pages at {self.dpi} DPI")
            logger.info(f"Images will be saved to: {output_path}")
            
            # Process each page
            for page_num, page in enumerate(doc, start=1):
                # Convert page to pixmap (image)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Generate filename
                image_filename = output_path / f"page_{page_num:04d}.png"
                
                # Save as PNG
                pix.save(str(image_filename))
                
                # Alternative: Save using PIL for more format options
                # img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # img.save(image_filename, "PNG", dpi=(self.dpi, self.dpi))
                
                image_paths.append(str(image_filename))
                logger.info(f"Converted page {page_num}/{len(doc)}")
            
            doc.close()
            logger.info(f"PDF conversion completed: {len(image_paths)} pages")
            
            return image_paths
            
        except Exception as e:
            logger.error(f"PyMuPDF conversion failed: {str(e)}")
            raise
    
    def convert_with_pdf2image(
        self, 
        pdf_path: str, 
        output_dir: str
    ) -> List[str]:
        """
        Convert PDF to images using pdf2image (Poppler wrapper).
        
        Requires Poppler to be installed on the system.
        - Ubuntu/Debian: sudo apt-get install poppler-utils
        - macOS: brew install poppler
        - Windows: Download from https://github.com/oschwartz10612/poppler-windows
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            
        Returns:
            List of image file paths
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert PDF to PIL Images
            logger.info(f"Converting PDF at {self.dpi} DPI using pdf2image")
            
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='png',
                thread_count=4  # Parallel processing
            )
            
            image_paths = []
            
            # Save each page
            for page_num, image in enumerate(images, start=1):
                image_filename = output_path / f"page_{page_num:04d}.png"
                image.save(image_filename, "PNG", dpi=(self.dpi, self.dpi))
                image_paths.append(str(image_filename))
                logger.info(f"Saved page {page_num}/{len(images)}")
            
            logger.info(f"PDF conversion completed: {len(image_paths)} pages")
            
            return image_paths
            
        except Exception as e:
            logger.error(f"pdf2image conversion failed: {str(e)}")
            raise
    
    def convert(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Convert PDF to images using configured method.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            
        Returns:
            List of image file paths
        """
        if self.method == "pymupdf":
            return self.convert_with_pymupdf(pdf_path, output_dir)
        elif self.method == "pdf2image":
            return self.convert_with_pdf2image(pdf_path, output_dir)
        else:
            raise ValueError(f"Unknown conversion method: {self.method}")
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get PDF metadata and information.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            doc = fitz.open(pdf_path)
            
            info = {
                'page_count': len(doc),
                'metadata': doc.metadata,
                'is_encrypted': doc.is_encrypted,
                'needs_pass': doc.needs_pass,
                'page_sizes': []
            }
            
            # Get size of each page
            for page in doc:
                rect = page.rect
                info['page_sizes'].append({
                    'width': rect.width,
                    'height': rect.height
                })
            
            doc.close()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get PDF info: {str(e)}")
            raise
    
    def extract_specific_pages(
        self, 
        pdf_path: str, 
        output_dir: str,
        page_numbers: List[int]
    ) -> List[str]:
        """
        Extract only specific pages from PDF.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            page_numbers: List of page numbers (1-indexed)
            
        Returns:
            List of image file paths
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            doc = fitz.open(pdf_path)
            image_paths = []
            
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in page_numbers:
                if page_num < 1 or page_num > len(doc):
                    logger.warning(f"Page {page_num} out of range, skipping")
                    continue
                
                # Get page (0-indexed)
                page = doc[page_num - 1]
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                image_filename = output_path / f"page_{page_num:04d}.png"
                pix.save(str(image_filename))
                image_paths.append(str(image_filename))
                
                logger.info(f"Extracted page {page_num}")
            
            doc.close()
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Page extraction failed: {str(e)}")
            raise