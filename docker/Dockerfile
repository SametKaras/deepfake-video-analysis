# 1️⃣ NVIDIA CUDA ve CUDNN içeren resmi Ubuntu tabanlı imajı kullanıyoruz
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2️⃣ Sistem bağımlılıklarını yükleyelim
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    python3-opencv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Çalışma dizinini oluştur
WORKDIR /app

# 4️⃣ Gerekli Python kütüphanelerini yükleyelim (spesifik versiyonlarla)
RUN pip3 install --no-cache-dir \
    torch>=1.7.0 \
    torchvision>=0.8.0 \
    opencv-python>=4.5.1.48 \
    pandas>=1.1.5 \
    numpy>=1.19.5 \
    Pillow>=8.0.0

# 5️⃣ Proje dosyalarını konteynere kopyala
COPY ./src /app/src

# 6️⃣ Model dosyası burada olacak
COPY ./src/model.pth /app/src/model.pth

# 7️⃣ Konteyner başladığında çalıştırılacak komut
CMD ["python3", "/app/src/main.py"]
