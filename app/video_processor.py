import cv2
from face_recognition import FaceRecognizer
from config import *

class VideoProcessor:
    def __init__(self):
        self.face_recognizer = FaceRecognizer()
        self.camera = None
        self.frame_count = 0  # Добавляем счетчик кадров

    def _initialize_camera(self):
        """Initialize video capture from stream."""
        try:
            self.camera = cv2.VideoCapture(STREAM_URL)
            if not self.camera.isOpened():
                raise Exception("Failed to open video stream")
        except Exception as e:
            print(f"Error initializing camera: {str(e)}")
            raise

    def _process_frame(self, frame):
        """Process a single frame for face detection and recognition."""
        try:
            # Resize frame
             # frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            # Detect faces
            faces_xyxy, confidences = self.face_recognizer.detect_faces(frame)
            
            if len(faces_xyxy) == 0:
                return frame
            
            # Extract face crops
            face_crops = []
            for x1, y1, x2, y2 in faces_xyxy.int().tolist():
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    face_crops.append(face_img)
            
            if not face_crops:
                return frame
            
            # Recognize faces
            recognition_results = self.face_recognizer.recognize_faces(face_crops)
            
            # Draw results on frame
            for (x1, y1, x2, y2), (name, score) in zip(
                faces_xyxy.int().tolist(),
                recognition_results
            ):
                color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({score:.2f})" if name != "UNKNOWN" else name
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame

    def process_stream(self):
        """Process video stream and yield frames."""
        try:
            if self.camera is None:
                self._initialize_camera()
            
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame from video stream")
                    break
                
                # Пропускаем каждый второй кадр
                self.frame_count += 1
                if self.frame_count % 3 == 0:
                    continue
                
                processed_frame = self._process_frame(frame)
                _, encoded_image = cv2.imencode('.jpg', processed_frame)
                yield encoded_image.tobytes()
                
        except Exception as e:
            print(f"Error in video stream processing: {str(e)}")
        finally:
            if self.camera is not None:
                self.camera.release()
                self.camera = None 