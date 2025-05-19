# 🚀 Advanced Real-time Object Detection with YOLOv5 & Flask

Experience cutting-edge object detection powered by YOLOv5, featuring a sleek dark-mode interface and interactive AI capabilities. This project combines the power of YOLOv5's state-of-the-art object detection with a modern, responsive web interface.

## ✨ Key Features

- 🔥 Real-time object detection using YOLOv5
- 🎨 Modern dark-mode UI with particle effects
- 🤖 Natural Language Query Processing
- 📊 Confidence-based object highlighting
- 🎯 Interactive bounding boxes with scanning effects
- 📱 Responsive design for all devices
- 🔍 Advanced fallback detection system
- 💫 Dynamic FPS counter and performance metrics

## 🛠️ Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam access
- Modern web browser

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd webcam_proje
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv5 model**
   ```bash
   python download_yolov5.py
   ```
   Or manually:
   - Download YOLOv5s model from [YOLOv5 releases](https://github.com/ultralytics/yolov5/releases)
   - Place in `yolov5/weights/yolov5s.pt`

## 🎮 Usage

1. **Launch the application**
   ```bash
   python app.py
   ```

2. **Access the interface**
   Open your browser and navigate to `http://127.0.0.1:5000`

3. **Interactive Features**
   - Click "Start Camera" to begin detection
   - Ask questions about detected objects
   - Toggle fullscreen mode for immersive experience
   - View confidence scores and object details
   - Experience the particle background effects

## 🌟 Advanced Features

- **Smart Object Detection**: Leverages YOLOv5's advanced architecture for superior detection accuracy
- **Natural Language Processing**: Ask questions about detected objects in plain English
- **Confidence Visualization**: Color-coded bounding boxes based on detection confidence
- **Dynamic UI**: Interactive elements with smooth animations and transitions
- **Fallback System**: Intelligent fallback detection when YOLOv5 is unavailable

## 🔧 Configuration

Create a `.env` file in the project root:
```env
YOLOV5_DIR=yolov5
MODEL_PATH=yolov5/weights/yolov5s.pt
PORT=5000
```

## 🛠️ Troubleshooting

- **Camera Access**: Ensure your webcam is connected and accessible
- **Model Loading**: Verify YOLOv5 model files are correctly placed
- **Dependencies**: Check `requirements.txt` for all required packages
- **Browser Support**: Use a modern browser for best experience

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- Flask web framework
- OpenCV for image processing
- All contributors and supporters

---

Made with ❤️ and powered by cutting-edge AI technology