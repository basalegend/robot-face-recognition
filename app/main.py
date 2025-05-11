from flask import Flask, Response, render_template
import threading
import time
from video_processor import VideoProcessor
from config import *

app = Flask(__name__)

# Переменные для хранения данных
last_frame = None
latest_jpeg = None
frame_lock = threading.Lock()

# Инициализируем модель один раз
video_processor = VideoProcessor()

# Флаг работы
is_processing = True

def capture_frames():
    """Функция фонового потока для захвата и обработки кадров"""
    global last_frame, latest_jpeg, is_processing

    for frame_data in video_processor.process_stream():
        with frame_lock:
            latest_jpeg = frame_data  # Сохраняем последний кадр
        if not is_processing:
            break

# Запускаем фоновый поток ДО запуска сервера
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        with frame_lock:
            if latest_jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_jpeg + b'\r\n\r\n')
        time.sleep(0.03)  # ~30 fps

if __name__ == '__main__':
    print(f"Starting server on {HOST}:{PORT}")
    try:
        app.run(host=HOST, port=PORT, threaded=True)
    finally:
        is_processing = False