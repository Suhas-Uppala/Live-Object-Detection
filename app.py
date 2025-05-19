import os
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import sys
import torch

# Load .env (optional)
load_dotenv()

app = Flask(__name__)

# YOLOv5 settings
YOLOV5_DIR = os.getenv("YOLOV5_DIR", "yolov5")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(YOLOV5_DIR, "weights/yolov5s.pt"))

# Global variables for model
model = None
classes = []
fallback_classes = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
           "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def initialize_model():
    """Initialize YOLOv5 model"""
    global model, classes
    
    # Check if YOLOv5 is installed
    yolov5_exists = os.path.exists(YOLOV5_DIR)
    model_exists = os.path.exists(MODEL_PATH)
    
    if not yolov5_exists:
        print(f"WARNING: YOLOv5 directory not found at {YOLOV5_DIR}")
    if not model_exists:
        print(f"WARNING: Model weights file not found at {MODEL_PATH}")
    
    try:
        if yolov5_exists:
            # Add YOLOv5 to the path if it exists
            if YOLOV5_DIR not in sys.path:
                sys.path.append(YOLOV5_DIR)
            
            if model_exists:
                # Load YOLOv5 model
                model = torch.hub.load(YOLOV5_DIR, 'custom', path=MODEL_PATH, source='local')
                model.conf = 0.5  # Confidence threshold
                classes = model.names
                print(f"YOLOv5 model initialized successfully with {len(classes)} classes")
            else:
                # Try loading from torch hub if local model not found
                try:
                    print("Trying to load YOLOv5 from torch hub...")
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                    model.conf = 0.5  # Confidence threshold
                    classes = model.names
                    print(f"YOLOv5 model loaded from torch hub with {len(classes)} classes")
                except Exception as e:
                    print(f"Failed to load from torch hub: {str(e)}")
                    model = None
        else:
            print("YOLOv5 directory not found, using fallback detection")
            model = None
            classes = fallback_classes
            
    except Exception as e:
        print(f"Error initializing YOLOv5 model: {str(e)}")
        model = None
        classes = fallback_classes
        print("Using fallback class names due to error")

# Initialize model on startup
initialize_model()

def detect_objects(image):
    """Detect objects in the given image using YOLOv5 or fallback method"""
    global model, classes
    
    if model is None:
        # Enhanced fallback method for better object detection when YOLOv5 is not available
        print("Using enhanced fallback detection method")
        
        # Convert to different color spaces for feature extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        detected_objects = []
        detected_boxes = []
        
        # 1. Face detection using Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Add detected faces
        for (x, y, w, h) in faces:
            detected_objects.append("Person (0.85)")
            detected_boxes.append([x, y, w, h])
        
        # 2. More specialized object detection using color and shape analysis
        
        # Apply adaptive thresholding for better feature detection
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours in the threshold image
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour to determine object type
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Skip very small contours
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Extract region of interest for further analysis
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
                
            # Analyze color (average HSV in the ROI)
            try:
                roi_hsv = hsv[y:y+h, x:x+w]
                avg_hue = np.mean(roi_hsv[:, :, 0])
                avg_sat = np.mean(roi_hsv[:, :, 1])
                avg_val = np.mean(roi_hsv[:, :, 2])
            except:
                continue
            
            # Detect specific objects based on shape and color characteristics
            
            # Pen detection: typically elongated shape
            if 2.5 < aspect_ratio < 12.0 and area > 1000 and area < 20000:
                detected_objects.append("Pen (0.75)")
                detected_boxes.append([x, y, w, h])
                continue
                
            # Phone detection: rectangular shape with specific aspect ratio
            if 1.5 < aspect_ratio < 2.2 and area > 10000:
                detected_objects.append("Phone (0.72)")
                detected_boxes.append([x, y, w, h])
                continue
                
            # Book/paper detection: rectangular, typically larger
            if 0.5 < aspect_ratio < 1.5 and area > 30000:
                detected_objects.append("Book/Paper (0.68)")
                detected_boxes.append([x, y, w, h])
                continue
                
            # Cup/glass detection: more vertical objects
            if 0.5 < aspect_ratio < 1.0 and 5000 < area < 30000:
                detected_objects.append("Cup (0.65)")
                detected_boxes.append([x, y, w, h])
                continue
                
            # Generic object detection for larger contours
            if area > 5000:
                # Calculate shape complexity (more complex = more points in the approximated contour)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) > 6:  # More complex shapes
                    detected_objects.append("Complex Object (0.60)")
                elif len(approx) > 4:  # Medium complexity
                    detected_objects.append("Object (0.55)")
                else:  # Simple shapes
                    detected_objects.append("Simple Object (0.50)")
                    
                detected_boxes.append([x, y, w, h])
        
        # If nothing meaningful was detected, add a generic object
        if not detected_objects:
            height, width = image.shape[:2]
            detected_objects.append("Unknown Object (0.40)")
            detected_boxes.append([int(width*0.25), int(height*0.25), int(width*0.5), int(height*0.5)])
            
        return detected_objects, detected_boxes
    
    # YOLOv5 object detection
    try:
        # Convert BGR to RGB for YOLOv5
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(rgb_image)
        
        # Process results
        detected_objects = []
        detected_boxes = []
        
        # Get prediction results
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_id = int(detection['class'])
            label = classes[class_id]
            
            detected_objects.append(f"{label} ({confidence:.2f})")
            detected_boxes.append([x1, y1, x2-x1, y2-y1])  # Format as [x, y, width, height]
        
        return detected_objects, detected_boxes
    except Exception as e:
        print(f"Error during YOLOv5 detection: {str(e)}")
        # Fallback to basic detection if YOLOv5 fails
        return ["Error in detection (0.10)"], [[0, 0, 100, 100]]

def answer_query(query, detected_objects):
    """
    Generate a response to the user's query based on the detected objects
    """
    if not detected_objects or len(detected_objects) == 0:
        return "I don't see any objects in the image."
    
    # Extract just the object names without confidence values
    object_names = [obj.split(" (")[0] for obj in detected_objects]
    
    # Count objects by type
    object_counts = {}
    for obj in object_names:
        if obj in object_counts:
            object_counts[obj] += 1
        else:
            object_counts[obj] = 1
    
    # Handle different types of queries
    query = query.lower()
    
    # Check for "how many" queries
    if "how many" in query:
        for obj in object_names:
            if obj.lower() in query:
                count = object_counts[obj]
                return f"I can see {count} {obj}{'s' if count > 1 else ''}."
        
        # If no specific object was mentioned
        total = len(detected_objects)
        return f"I can see {total} object{'s' if total > 1 else ''} in total."
    
    # Check for "is there" or "do you see" queries
    if any(phrase in query for phrase in ["is there", "are there", "do you see", "can you see"]):
        for obj_type in object_counts.keys():
            if obj_type.lower() in query:
                count = object_counts[obj_type]
                return f"Yes, I can see {count} {obj_type}{'s' if count > 1 else ''}."
        
        # If asking for an object not found
        for cls in classes:
            if isinstance(cls, str) and cls.lower() in query:
                return f"No, I don't see any {cls} in the image."
    
    # Check for "what" queries
    if "what" in query and any(word in query for word in ["objects", "things", "do you see"]):
        if object_names:
            if len(object_names) == 1:
                return f"I can see a {object_names[0]}."
            else:
                formatted_objects = ", ".join(object_names[:-1]) + " and " + object_names[-1]
                return f"I can see: {formatted_objects}."
    
    # Default response
    return f"I can see the following objects: {', '.join(object_names)}."

@app.route("/")
def index():
    model_status = "loaded" if model is not None else "not_loaded"
    return render_template("index.html", model_status=model_status)

@app.route("/model_status")
def model_status():
    """Return the current status of the model including which files are missing"""
    try:
        model_exists = os.path.exists(MODEL_PATH)
        
        result = {
            "model_loaded": model is not None,
            "files_status": {
                "weights": {"path": MODEL_PATH, "exists": model_exists}
            },
            "message": "Model is ready" if model is not None else "Model files are missing. Using fallback detection.",
            "download_urls": {
                "weights": "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt"
            }
        }
        return jsonify(result)
    except Exception as e:
        print(f"Error in model_status route: {str(e)}")
        # If anything goes wrong, return a valid JSON error response
        return jsonify({
            "error": str(e),
            "model_loaded": False,
            "message": "Error checking model status"
        })

@app.route("/detect", methods=["POST"])
def detect():
    """
    Receives a JSON payload: { image: "<base64‑data>" }
    Processes the image for object detection using OpenCV,
    returns the detected objects back to the client.
    """
    data = request.get_json()
    img_b64 = data.get("image", "").split(",")[-1]
    
    try:
        # Convert base64 to image
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Detect objects with proper error handling
        try:
            detected_objects, detected_boxes = detect_objects(img)
        except Exception as e:
            # Fallback if unpacking fails
            print(f"Error during object detection: {str(e)}")
            detected_objects = []
            detected_boxes = []
        
        result = {
            "objects": detected_objects if detected_objects else [],
            "boxes": detected_boxes if detected_boxes else [],
            "count": len(detected_objects) if detected_objects else 0
        }
        
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def process_query():
    """
    Receives a JSON payload: { image: "<base64-data>", query: "<user-query>" }
    Processes the image for object detection and responds to the user's query
    based on the detected objects.
    """
    data = request.get_json()
    img_b64 = data.get("image", "").split(",")[-1]
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Convert base64 to image
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Detect objects with proper error handling
        try:
            detected_objects, detected_boxes = detect_objects(img)
        except Exception as e:
            # Fallback if unpacking fails
            print(f"Error during object detection: {str(e)}")
            detected_objects = []
            detected_boxes = []
        
        # Process the query against detected objects
        answer = answer_query(query, detected_objects)
        
        result = {
            "objects": detected_objects if detected_objects else [],
            "boxes": detected_boxes if detected_boxes else [],
            "count": len(detected_objects) if detected_objects else 0
        }
        
        return jsonify({
            "result": result,
            "answer": answer,
            "query": query
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
