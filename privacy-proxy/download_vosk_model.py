#!/usr/bin/env python3
"""Download Vosk model for speech recognition"""
import subprocess
import sys
from pathlib import Path
import zipfile

def download_vosk_model():
    """Download and extract Vosk English model"""
    model_name = "vosk-model-en-us-0.22-lgraph"
    model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"

    print(f"Downloading Vosk model: {model_name}")
    print(f"URL: {model_url}")
    print("This may take a minute...")

    # Check if already exists
    if Path(model_name).exists():
        print(f"\n✓ Model already exists: {model_name}")
        return True

    # Download
    try:
        subprocess.run([
            "curl", "-L", "-o", f"{model_name}.zip", model_url
        ], check=True)

        print("\n✓ Download complete!")

        # Extract
        print("Extracting model...")
        with zipfile.ZipFile(f"{model_name}.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

        print("✓ Extraction complete!")

        # Cleanup zip file
        Path(f"{model_name}.zip").unlink()

        print(f"\n✓ Model ready at: ./{model_name}")
        print(f"Model size: {sum(f.stat().st_size for f in Path(model_name).rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")

        return True

    except subprocess.CalledProcessError:
        print("\n✗ Download failed!")
        print("Please download manually from:")
        print(f"  {model_url}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = download_vosk_model()
    sys.exit(0 if success else 1)
