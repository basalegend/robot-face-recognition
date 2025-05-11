import os

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'app', 'models', 'best.pt')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'app', 'models', 'embeddings.pt')

# Detection settings
DETECTION_THRESHOLD = 0.5
RECOGNITION_THRESHOLD = 0.4

# Video settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FACE_SIZE = 160

# Server settings
HOST = '0.0.0.0'
PORT = 8000
DEBUG = False

# Stream settings
STREAM_URL = 'http://172.20.10.12:5000/video_feed' 