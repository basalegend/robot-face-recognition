import cv2
import torch
import numpy as np
import torchvision
from torchvision import transforms
from ultralytics import YOLO

# Параметры
YOLO_MODEL_PATH = '../app/models/best.pt'
RESNET_MODEL_PATH = '../face_recognition_resnet.pth'
CLASS_NAMES_PATH = 'labels_organized.txt'
CONFIDENCE_THRESHOLD = 0.4
IMG_SIZE = 224

# Загрузка YOLO модели
yolo_model = YOLO(YOLO_MODEL_PATH)

# Загрузка ResNet18 модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet_model = torchvision.models.resnet18(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_ftrs, len(open(CLASS_NAMES_PATH).readlines()))  # число классов
resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
resnet_model.to(device)
resnet_model.eval()

# Преобразования для ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка имён классов
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Захват с камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обнаружение лиц с помощью YOLO
    results = yolo_model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)

            # Вырезаем лицо
            face_img = frame[y1:y2, x1:x2]
            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                continue

            # Подготовка к ResNet
            input_tensor = transform(face_img).unsqueeze(0).to(device)

            # Распознавание
            with torch.no_grad():
                outputs = resnet_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence_resnet, predicted = torch.max(probs, 1)
                confidence_resnet = confidence_resnet.item()
                name = class_names[predicted.item()] if confidence_resnet >= CONFIDENCE_THRESHOLD else "UNKNOWN"

            # Рисуем bounding box и подпись
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()