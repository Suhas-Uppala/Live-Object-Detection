import os
import sys
import urllib.request
import time

def download_file(url, filename):
    """
    Download a file from a URL showing progress
    """
    try:
        if os.path.exists(filename):
            print(f"{filename} already exists. Skipping download.")
            return True
            
        print(f"Downloading {url} to {filename}...")
        
        def report_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r{filename}: {percent}% complete")
            sys.stdout.flush()
            
        start_time = time.time()
        urllib.request.urlretrieve(url, filename, reporthook=report_progress)
        end_time = time.time()
        
        print(f"\nDownload completed in {end_time - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False

def main():
    """
    Download the YOLO model files needed for object detection
    """
    print("=== YOLO Model Downloader ===")
    print("This script will download the necessary YOLOv3 model files for object detection.")
    print("Files will be saved in the current directory.")
    print("Total download size: ~240MB\n")
    
    files_to_download = [
        ("https://pjreddie.com/media/files/yolov3.weights", "yolov3.weights"),
        ("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", "yolov3.cfg"),
        ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", "coco.names")
    ]
    
    success = True
    for url, filename in files_to_download:
        if not download_file(url, filename):
            success = False
            
    if success:
        print("\nAll model files downloaded successfully!")
        print("You can now restart the application to use the YOLO object detection model.")
    else:
        print("\nSome downloads failed. Please check the errors above and try again.")
        print("You can also download the files manually from:")
        print("- YOLOv3 weights: https://pjreddie.com/media/files/yolov3.weights")
        print("- YOLOv3 config: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg")
        print("- COCO classes: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
        
if __name__ == "__main__":
    main() 