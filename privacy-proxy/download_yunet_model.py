#!/usr/bin/env python3
"""Download YuNet face detection model from OpenCV Zoo"""

import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH = Path("face_detection_yunet_2023mar.onnx")

def download_model():
    if MODEL_PATH.exists():
        logger.info(f"Model already exists at {MODEL_PATH}")
        return True
    
    logger.info(f"Downloading YuNet model from {MODEL_URL}")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    logger.info(f"Progress: {percent:.1f}%")
        
        logger.info(f"Successfully downloaded model to {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

if __name__ == "__main__":
    if download_model():
        logger.info("YuNet model is ready to use!")
    else:
        logger.error("Failed to download YuNet model")