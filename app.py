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
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
           # Additional specific objects
           "pen", "pencil", "marker", "notebook", "folder", "paper", "envelope", "stamp", "stapler",
           "paperclip", "rubber band", "tape", "glue", "scissors", "ruler", "calculator", "headphones",
           "earphones", "charger", "cable", "usb drive", "memory card", "camera", "tripod", "microphone",
           "speaker", "printer", "scanner", "monitor", "keyboard", "mouse pad", "desk lamp", "coffee mug",
           "water bottle", "lunch box", "snack", "candy", "chips", "cookies", "fruit", "vegetables",
           "bread", "cereal", "milk", "juice", "soda", "beer", "wine", "coffee", "tea", "spices",
           "salt", "pepper", "sugar", "honey", "jam", "butter", "cheese", "yogurt", "meat", "fish",
           "chicken", "eggs", "rice", "pasta", "sauce", "soup", "salad", "sandwich", "burger", "fries",
           "pizza slice", "taco", "sushi", "noodles", "dumplings", "cake slice", "ice cream", "chocolate",
           "candy bar", "gum", "mints", "vitamins", "medicine", "bandage", "cotton", "tissue", "napkin",
           "toilet paper", "soap", "shampoo", "conditioner", "toothpaste", "deodorant", "perfume",
           "makeup", "brush", "comb", "mirror", "towel", "washcloth", "sponge", "cleaning spray",
           "broom", "mop", "vacuum", "dustpan", "trash can", "recycling bin", "plant", "flower",
           "candle", "picture frame", "painting", "poster", "calendar", "clock", "watch", "jewelry",
           "necklace", "bracelet", "ring", "earrings", "glasses", "sunglasses", "hat", "scarf",
           "gloves", "socks", "shoes", "boots", "sandals", "slippers", "jacket", "coat", "sweater",
           "shirt", "pants", "shorts", "skirt", "dress", "suit", "tie", "belt", "wallet", "purse",
           "backpack", "briefcase", "luggage", "umbrella", "cane", "walking stick", "camera", "binoculars",
           "telescope", "compass", "map", "guidebook", "ticket", "passport", "credit card", "money",
           "coins", "keys", "phone case", "tablet case", "laptop case", "camera case", "guitar case",
           "instrument", "music sheet", "bookmark", "magnifying glass", "flashlight", "batteries",
           "extension cord", "power strip", "surge protector", "router", "modem", "antenna", "satellite dish",
           "solar panel", "wind turbine", "generator", "battery", "fuel", "oil", "gas", "water",
           "fire", "smoke", "steam", "cloud", "rain", "snow", "ice", "fog", "mist", "dew", "frost",
           "lightning", "thunder", "wind", "storm", "hurricane", "tornado", "earthquake", "volcano",
           "mountain", "hill", "valley", "river", "lake", "ocean", "beach", "desert", "forest",
           "jungle", "grassland", "meadow", "garden", "park", "playground", "stadium", "arena",
           "theater", "cinema", "museum", "library", "school", "university", "hospital", "clinic",
           "pharmacy", "store", "market", "mall", "restaurant", "cafe", "bar", "hotel", "motel",
           "house", "apartment", "building", "skyscraper", "bridge", "tunnel", "road", "street",
           "highway", "railway", "airport", "port", "station", "terminal", "gate", "door", "window",
           "wall", "floor", "ceiling", "roof", "stairs", "elevator", "escalator", "ramp", "path",
           "trail", "track", "field", "court", "pool", "fountain", "statue", "monument", "tower",
           "antenna", "satellite", "rocket", "spacecraft", "satellite", "planet", "star", "moon",
           "sun", "galaxy", "universe"]

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
    Generate a detailed and contextual response to any user query based on the detected objects
    """
    if not detected_objects or len(detected_objects) == 0:
        return "I don't see any objects in the current frame."

    # Extract object names and confidence values
    object_details = []
    for obj in detected_objects:
        if " (" in obj:
            name, conf = obj.split(" (")
            conf = float(conf.rstrip(")"))
            object_details.append({"name": name, "confidence": conf})
        else:
            object_details.append({"name": obj, "confidence": 1.0})

    # Count objects by type and calculate statistics
    object_counts = {}
    object_confidences = {}
    for obj in object_details:
        name = obj["name"]
        if name in object_counts:
            object_counts[name] += 1
            object_confidences[name].append(obj["confidence"])
        else:
            object_counts[name] = 1
            object_confidences[name] = [obj["confidence"]]

    # Calculate average confidences
    avg_confidences = {
        name: sum(confs) / len(confs) 
        for name, confs in object_confidences.items()
    }

    # Convert query to lowercase for easier matching
    query = query.lower()

    # Handle different types of questions

    # 1. Questions about specific objects
    for obj_type in object_counts:
        if obj_type.lower() in query:
            count = object_counts[obj_type]
            avg_conf = avg_confidences[obj_type]
            
            # Handle different question types for specific objects
            if "where" in query:
                return f"I can see {count} {obj_type}{'s' if count > 1 else ''} in the frame."
            elif "how many" in query:
                return f"There {'are' if count > 1 else 'is'} {count} {obj_type}{'s' if count > 1 else ''} in the frame."
            elif any(word in query for word in ["is there", "are there", "do you see", "can you see"]):
                return f"Yes, I can see {count} {obj_type}{'s' if count > 1 else ''} with {avg_conf:.0%} confidence."
            else:
                return f"I can see {count} {obj_type}{'s' if count > 1 else ''} in the frame with {avg_conf:.0%} confidence."

    # 2. Questions about the scene/video
    if any(word in query for word in ["video", "frame", "scene", "what's happening", "what do you see"]):
        # Sort objects by confidence
        sorted_objects = sorted(object_details, key=lambda x: x["confidence"], reverse=True)
        
        # Create a natural language description
        if len(sorted_objects) == 1:
            return f"In this frame, I can see a {sorted_objects[0]['name']} with {sorted_objects[0]['confidence']:.0%} confidence."
        
        # Group objects by type for better description
        grouped_objects = {}
        for obj in sorted_objects:
            name = obj["name"]
            if name not in grouped_objects:
                grouped_objects[name] = []
            grouped_objects[name].append(obj["confidence"])

        descriptions = []
        for obj_type, confidences in grouped_objects.items():
            count = len(confidences)
            avg_conf = sum(confidences) / len(confidences)
            if count > 1:
                descriptions.append(f"{count} {obj_type}s")
            else:
                descriptions.append(f"a {obj_type}")

        if len(descriptions) == 1:
            return f"In this frame, I can see {descriptions[0]}."
        else:
            return f"In this frame, I can see {', '.join(descriptions[:-1])}, and {descriptions[-1]}."

    # 3. Questions about confidence or certainty
    if any(word in query for word in ["confident", "certain", "sure", "accurate"]):
        sorted_objects = sorted(object_details, key=lambda x: x["confidence"], reverse=True)
        most_confident = sorted_objects[0]
        return f"I am most confident about detecting a {most_confident['name']} with {most_confident['confidence']:.0%} confidence."

    # 4. Questions about the environment or setting
    if any(word in query for word in ["environment", "setting", "place", "location"]):
        # Group objects by category (you can expand these categories)
        categories = {
            "people": ["person", "people", "man", "woman", "child"],
            "furniture": ["chair", "table", "desk", "sofa", "bed"],
            "electronics": ["laptop", "phone", "tv", "monitor", "camera"],
            "objects": ["book", "pen", "paper", "bag", "cup"]
        }
        
        detected_categories = {}
        for obj in object_details:
            for category, items in categories.items():
                if obj["name"].lower() in items:
                    if category not in detected_categories:
                        detected_categories[category] = []
                    detected_categories[category].append(obj["name"])

        if detected_categories:
            category_descriptions = []
            for category, items in detected_categories.items():
                if len(items) > 1:
                    category_descriptions.append(f"some {category}")
                else:
                    category_descriptions.append(f"a {items[0]}")
            
            return f"This appears to be a setting with {', '.join(category_descriptions)}."
        else:
            return "I can see various objects in the frame, but I'm not sure about the specific setting."

    # 5. Questions about movement or activity
    if any(word in query for word in ["moving", "activity", "action", "doing"]):
        if "person" in object_counts:
            return "I can see a person in the frame, but I can't determine their specific activity from a single frame."
        else:
            return "I don't see any people in the frame to describe their activity."

    # 6. Questions about relationships between objects
    if any(word in query for word in ["relationship", "between", "relative", "position"]):
        if len(object_details) > 1:
            return f"I can see multiple objects in the frame, but I can't determine their spatial relationships from a single frame."
        else:
            return "I only see one object in the frame, so there are no relationships to describe."

    # 7. Questions about time or duration
    if any(word in query for word in ["time", "duration", "long", "how long"]):
        return "I can only analyze individual frames, so I can't determine time-related information."

    # Default response for other questions
    sorted_objects = sorted(object_details, key=lambda x: x["confidence"], reverse=True)
    formatted_objects = ", ".join([f"{obj['name']} ({obj['confidence']:.0%})" for obj in sorted_objects])
    return f"In this frame, I can see: {formatted_objects}."

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
