from facenet_pytorch import InceptionResnetV1
import torch
import os
import cv2
import time
from sklearn.metrics.pairwise import cosine_similarity

DATASET_DIR = "../label-studio/label-studio-files/cropped_dataset"
EMBEDDINGS_PATH = "app/models/embeddings.pt"

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).div(255).to(device)
    with torch.no_grad():
        embedding = model(img).cpu()
    return embedding

# Создание базы эмбеддингов
def build_embeddings_db():
    db = {}
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for file in os.listdir(person_dir):
            emb = get_embedding(os.path.join(person_dir, file))
            embeddings.append(emb)
        db[person_name] = torch.mean(torch.vstack(embeddings), dim=0)
    torch.save(db, EMBEDDINGS_PATH)
    print("База эмбеддингов создана.")

# Распознавание
def recognize_face(image_path, threshold=0.6):
    start = time.time()
    emb = get_embedding(image_path)
    best_match = "UNKNOWN"
    best_score = 0
    for name, db_emb in db.items():
        score = cosine_similarity(emb, db_emb.unsqueeze(0)).item()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = name
    end = time.time()
    print(end-start)
    return best_match, best_score

# === Пример использования ===
build_embeddings_db()
