# Real-time Webcam Object Detection with Flask

This project uses Flask and OpenCV to perform real-time object detection on webcam feeds.

## Prerequisites

1. Install Python 3.8 or higher.
2. Install pip (Python package manager).
3. Install virtualenv (optional but recommended).

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd webcam_proje
   ```

2. **Create a virtual environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv3 model files**
   - Download `yolov3.weights` from [YOLO official website](https://pjreddie.com/darknet/yolo/).
   - Download `yolov3.cfg` and `coco.names` from the same source.
   - Place these files in the project directory or update the `.env` file with their paths.

5. **Set up environment variables**
   Create a `.env` file in the project root with the following content:
   ```env
   MODEL_PATH=yolov3.weights
   CONFIG_PATH=yolov3.cfg
   CLASSES_PATH=coco.names
   PORT=5000
   ```

## Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage

- Click "Start Camera" to begin object detection.
- Detected objects and bounding boxes will be displayed in real-time.

## Troubleshooting

- Ensure your webcam is connected and accessible.
- Verify that the YOLO model files are correctly placed and paths are set in the `.env` file.
- Check for missing Python dependencies and install them using `pip install -r requirements.txt`.

## License

This project is licensed under the MIT License.