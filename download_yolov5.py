import os
import sys
import subprocess
import requests
from pathlib import Path

def download_file(url, save_path):
    """Download a file from a URL to the specified path"""
    print(f"Downloading from {url} to {save_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    downloaded = 0
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            downloaded += len(data)
            file.write(data)
            
            # Print progress
            progress = downloaded / total_size * 100
            status = f"\rDownloading: [{downloaded}/{total_size}] {progress:.2f}% "
            sys.stdout.write(status)
            sys.stdout.flush()
    
    print("\nDownload completed successfully")

def main():
    """Main function to set up YOLOv5"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yolov5_dir = os.path.join(current_dir, "yolov5")
    weights_path = os.path.join(yolov5_dir, "weights", "yolov5s.pt")
    
    # Step 1: Clone YOLOv5 repository if not exists
    if not os.path.exists(yolov5_dir):
        print("Cloning YOLOv5 repository...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git", yolov5_dir])
        print("YOLOv5 repository cloned successfully")
    else:
        print("YOLOv5 repository already exists, skipping clone")
    
    # Step 2: Install YOLOv5 requirements
    requirements_path = os.path.join(yolov5_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        print("Installing YOLOv5 requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("YOLOv5 requirements installed successfully")
    else:
        print("YOLOv5 requirements.txt not found at expected location")
    
    # Step 3: Download YOLOv5s weights if not exists
    if not os.path.exists(weights_path):
        print("Downloading YOLOv5s weights...")
        weights_url = "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt"
        download_file(weights_url, weights_path)
    else:
        print("YOLOv5s weights already exist, skipping download")
    
    print("\nSetup complete!")
    print(f"YOLOv5 installed at: {yolov5_dir}")
    print(f"YOLOv5s weights located at: {weights_path}")
    print("\nYou can now use YOLOv5 in your application!")

if __name__ == "__main__":
    main() 