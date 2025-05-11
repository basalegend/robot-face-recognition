FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Установка системных зависимостей
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app/main.py"]