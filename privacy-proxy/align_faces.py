#!/usr/bin/env python3
"""
Face alignment and quality control script for allowlist preparation.
Processes images to ensure high-quality face embeddings.
"""

import cv2
import face_recognition
import numpy as np
from pathlib import Path
import logging
import shutil
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_image_quality(img: np.ndarray, img_path: Path) -> Tuple[bool, str]:
    """
    Check if image meets quality standards for face recognition.
    
    Returns:
        (passed, reason) - True if image passes all checks, False with reason if not
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check 1: Sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False, f"Image too blurry (sharpness={laplacian_var:.1f}, need >100)"
    
    # Check 2: Brightness
    mean_brightness = gray.mean()
    if mean_brightness < 50 or mean_brightness > 200:
        return False, f"Poor lighting (brightness={mean_brightness:.0f}, need 50-200)"
    
    # Check 3: Exactly one face
    face_locations = face_recognition.face_locations(img, model="hog")
    if len(face_locations) == 0:
        return False, "No face detected"
    if len(face_locations) > 1:
        return False, f"Multiple faces detected ({len(face_locations)})"
    
    # Check 4: Resolution
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < 256:
        return False, f"Resolution too low ({w}x{h}, need min 256px)"
    
    return True, "All checks passed"


def align_and_crop_face(img: np.ndarray, output_size: int = 256) -> Optional[np.ndarray]:
    """
    Align and crop face to standard size using face_recognition landmarks.
    Includes padding to capture full face with hair and face outline.
    
    Returns:
        Aligned face image or None if alignment fails
    """
    # Find face locations and landmarks
    face_locations = face_recognition.face_locations(img, model="hog")
    if not face_locations:
        return None
    
    face_landmarks_list = face_recognition.face_landmarks(img, face_locations)
    if not face_landmarks_list:
        return None
    
    # Get the first face
    top, right, bottom, left = face_locations[0]
    face_landmarks = face_landmarks_list[0]
    
    # Calculate face dimensions
    face_width = right - left
    face_height = bottom - top
    
    # Add generous padding (50% on each side for hair and face outline)
    padding_ratio = 0.5
    pad_x = int(face_width * padding_ratio)
    pad_y = int(face_height * padding_ratio)
    
    # Calculate padded bounds
    padded_left = max(0, left - pad_x)
    padded_right = min(img.shape[1], right + pad_x)
    padded_top = max(0, top - pad_y)
    padded_bottom = min(img.shape[0], bottom + pad_y)
    
    # Calculate eye centers for rotation
    left_eye_pts = face_landmarks['left_eye']
    right_eye_pts = face_landmarks['right_eye']
    
    left_eye_center = np.mean(left_eye_pts, axis=0)
    right_eye_center = np.mean(right_eye_pts, axis=0)
    
    # Calculate angle for alignment
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Get the center of the padded region
    center_x = (padded_left + padded_right) / 2
    center_y = (padded_top + padded_bottom) / 2
    
    # Create rotation matrix around the center
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # First rotate the entire image
    h, w = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (w, h), 
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    
    # Transform the padded bounds
    corners = np.array([
        [padded_left, padded_top, 1],
        [padded_right, padded_top, 1],
        [padded_right, padded_bottom, 1],
        [padded_left, padded_bottom, 1]
    ]).T
    
    rotated_corners = M @ corners
    
    # Get new bounds after rotation
    min_x = int(np.min(rotated_corners[0]))
    max_x = int(np.max(rotated_corners[0]))
    min_y = int(np.min(rotated_corners[1]))
    max_y = int(np.max(rotated_corners[1]))
    
    # Ensure bounds are within image
    min_x = max(0, min_x)
    max_x = min(w, max_x)
    min_y = max(0, min_y)
    max_y = min(h, max_y)
    
    # Crop the rotated region
    cropped = rotated[min_y:max_y, min_x:max_x]
    
    # Resize to output size while maintaining aspect ratio
    crop_h, crop_w = cropped.shape[:2]
    if crop_h > crop_w:
        new_h = output_size
        new_w = int(crop_w * (output_size / crop_h))
    else:
        new_w = output_size
        new_h = int(crop_h * (output_size / crop_w))
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Create final image with padding to make it square
    final = np.full((output_size, output_size, 3), 128, dtype=np.uint8)
    y_offset = (output_size - new_h) // 2
    x_offset = (output_size - new_w) // 2
    final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return final


def process_allowlist_images(
    input_dir: Path = Path("allowlist_raw"),
    output_dir: Path = Path("allowlist"),
    align: bool = True,
    check_quality: bool = True,
    backup: bool = True
):
    """
    Process all images in the input directory for use in face allowlist.
    
    Args:
        input_dir: Directory containing raw face images
        output_dir: Directory to save processed images
        align: Whether to align faces
        check_quality: Whether to enforce quality checks
        backup: Whether to backup original images
    """
    if not input_dir.exists():
        logger.error(f"Input directory '{input_dir}' does not exist")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Statistics
    processed = 0
    failed = 0
    
    # Process each person's directory
    for person_dir in input_dir.iterdir():
        if not person_dir.is_dir():
            continue
            
        person_name = person_dir.name
        output_person_dir = output_dir / person_name
        output_person_dir.mkdir(exist_ok=True)
        
        logger.info(f"\nProcessing images for: {person_name}")
        
        # Process each image
        for img_path in person_dir.glob("*.[jp][pn]g"):
            logger.info(f"  Processing: {img_path.name}")
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"    Failed to load image")
                failed += 1
                continue
            
            # Quality check
            if check_quality:
                passed, reason = check_image_quality(img, img_path)
                if not passed:
                    logger.warning(f"    Quality check failed: {reason}")
                    failed += 1
                    continue
            
            # Align face if requested
            if align:
                aligned = align_and_crop_face(img)
                if aligned is None:
                    logger.warning(f"    Failed to align face")
                    failed += 1
                    continue
                img = aligned
                logger.info(f"    Face aligned and cropped to 256x256")
            
            # Save processed image
            output_path = output_person_dir / img_path.name
            
            # Backup original if it exists
            if output_path.exists() and backup:
                backup_path = output_path.with_suffix('.backup' + output_path.suffix)
                shutil.copy2(output_path, backup_path)
                logger.info(f"    Backed up existing image to {backup_path.name}")
            
            # Save with high quality
            if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(output_path), img)
            
            logger.info(f"    ✓ Saved to: {output_path}")
            processed += 1
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing complete!")
    logger.info(f"  Processed: {processed} images")
    logger.info(f"  Failed: {failed} images")
    logger.info(f"  Output directory: {output_dir}")
    
    if failed > 0:
        logger.warning(f"\n{failed} images failed processing. Check the logs above for details.")
        logger.warning("Common issues: blurry images, poor lighting, no face detected, or multiple faces")


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Align and validate faces for allowlist enrollment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images from allowlist_raw/ to allowlist/
  python align_faces.py
  
  # Process from custom directory without alignment
  python align_faces.py --input photos/ --no-align
  
  # Skip quality checks for testing
  python align_faces.py --no-quality-check
  
Directory structure:
  input_dir/
  ├── Alice Smith/
  │   ├── photo1.jpg
  │   └── photo2.jpg
  └── Bob Lee/
      └── photo1.jpg
""")
    
    parser.add_argument("--input", "-i", type=Path, default=Path("allowlist_raw"),
                        help="Input directory containing person folders (default: allowlist_raw/)")
    parser.add_argument("--output", "-o", type=Path, default=Path("allowlist"),
                        help="Output directory for processed images (default: allowlist/)")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip face alignment")
    parser.add_argument("--no-quality-check", action="store_true",
                        help="Skip quality checks")
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't backup existing images")
    
    args = parser.parse_args()
    
    process_allowlist_images(
        input_dir=args.input,
        output_dir=args.output,
        align=not args.no_align,
        check_quality=not args.no_quality_check,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()