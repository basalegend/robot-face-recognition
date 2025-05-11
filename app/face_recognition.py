import torch
import cv2
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from config import *


class FaceRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()
        self._load_embeddings()

    def _initialize_models(self):
        """Initialize YOLO and FaceNet models."""
        try:
            self.model_yolo = YOLO(YOLO_MODEL_PATH)
            self.model_yolo.eval().to(self.device)
            self.model_facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    def _load_embeddings(self):
        """Load face embeddings database."""
        try:
            self.embeddings_db = torch.load(EMBEDDINGS_PATH)
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            raise

    def get_face_embeddings_batch(self, face_imgs):
        """Get embeddings for a batch of face images."""
        face_tensors = []
        for img in face_imgs:
            img_resized = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
            tensor = torch.tensor(img_resized).float().permute(2, 0, 1).div(255).to(self.device)
            face_tensors.append(tensor)

        batch_tensor = torch.stack(face_tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model_facenet(batch_tensor)
        return embeddings

    def detect_faces(self, frame):
        """Detect faces in a frame using YOLO."""
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.model_yolo(img_pil)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            return [], []

        faces_xyxy = boxes.xyxy
        confidences = boxes.conf
        
        # Filter by confidence threshold
        valid_indices = confidences >= DETECTION_THRESHOLD
        faces_xyxy = faces_xyxy[valid_indices]
        confidences = confidences[valid_indices]
        
        return faces_xyxy, confidences

    def recognize_faces(self, face_crops):
        """Recognize faces using FaceNet embeddings."""
        if not face_crops:
            return []

        embeddings = self.get_face_embeddings_batch(face_crops)
        
        # Get embeddings database as tensor
        db_names = list(self.embeddings_db.keys())
        db_embs = torch.stack([emb.to(self.device) for emb in self.embeddings_db.values()])
        
        # Calculate cosine similarity
        sim_matrix = torch.mm(embeddings, db_embs.T)
        best_scores, best_indices = sim_matrix.max(dim=1)
        
        # Determine names
        recognized_names = [
            db_names[i] if score >= RECOGNITION_THRESHOLD else "UNKNOWN"
            for i, score in zip(best_indices, best_scores)
        ]
        
        return list(zip(recognized_names, best_scores.tolist())) 