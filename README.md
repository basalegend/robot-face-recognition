# Распознаванием лиц роботом-андроидом

**Мониторинг робота с использованием видеопотока и распознавания лиц через YOLO + FaceNet.**

Этот проект предоставляет **веб-интерфейс мониторинга**, который:
- Получает видеопоток с камеры
- Обнаруживает лица в реальном времени
- Распознаёт лица по базе данных эмбеддингов
- Отображает информацию о состоянии системы
- Может быть запущен в контейнере с поддержкой GPU

## Технологии

- Python 3.12
- Flask — веб-сервер
- PyTorch — для модели FaceNet
- YOLOv8 — для детекции лиц
- Facenet-PyTorch — реализация FaceNet
- OpenCV — работа с кадрами
- CUDA / GPU — ускорение инференса
- Docker — контейнеризация
- HTML/CSS/JS — интерфейс пользователя

---

## Структура Проекта

```
├── app/
│   ├── main.py                # Основной сервер Flask
│   ├── video_processor.py     # Обработка кадров и инференс модели
│   ├── config.py              # Конфигурационные параметры
│   ├── face_recognition.py    # Логика распознавания лиц
│   └── models/                # Основной сервер Flask
│       ├── best.pt            # Веса YOLO-модели
│       ├── embeddings.pt      # БД эмбэддингов лиц
│       └── facenet_model.pt   # Структура facenet-модели
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css      # Стили веб-интерфейса
│   │   └── js/
│   │       └── main.js        # JS-логика интерфейса
│   └── templates/
│       └── index.html         # Шаблон главной страницы
├── requirements.txt           # Зависимости Python
├── Dockerfile                 # Описание образа Docker
├── docker-compose.yml         # Настройки запуска контейнера
└── facenet.py                 # Получение эмбэддингов лиц
```

---

## Требования

### Аппаратные требования:
- Совместимая веб-камера и веб-сервер с ней
- NVIDIA GPU (если используется GPU-ускорение)

### Программные требования:
- Linux (Ubuntu 20.04 или выше)
- Docker + Docker Compose
- NVIDIA Driver (для GPU)
- NVIDIA Container Toolkit (для GPU)

---

## Установка и Запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/robot-monitoring.git
cd robot-monitoring
```

### 2. Установка зависимостей

Для запуска необходим установленный **Docker** и **Docker Compose**.

#### Установка Docker:

```bash
sudo apt update && sudo apt install docker.io -y
```

#### Установка Docker Compose:

```bash
sudo apt install docker-compose -y
```

#### Установка NVIDIA Container Toolkit (если используется GPU):

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

### 3. Сборка и запуск контейнера

```bash
docker-compose up --build
```

> Это соберёт образ и запустит приложение на порту `8000`.

Откройте браузер и перейдите по адресу:  
👉 http://localhost:8000


## Настройка работы с GPU

Чтобы использовать GPU, необходимо:

1. Убедиться, что драйверы NVIDIA установлены корректно.
2. Установлен `nvidia-container-toolkit`.
3. В `docker-compose.yml` указана секция:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Разработка

При разработке удобнее запускать приложение без Docker:

### 1. Установите зависимости:

```bash
pip install -r requirements.txt
```

### 2. Запустите сервер:

```bash
python app/main.py
```

> При таком способе учтите, что модель будет использовать GPU, если он доступен.

---
