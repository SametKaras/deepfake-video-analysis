# **3.2.3. Görev: Mimiklerin analiz edilmesi**

Bu proje, **CİNGÖZ Yarışması Etap-1** kapsamında **mimiklerin analiz edilmesi** amacıyla geliştirilmiştir.  
Proje, **CNN** tabanlı görüntü analizi gibi teknikleri kullanarak mimiklerin analiz tespit etmektedir.

---

## **📌 1. Gereksinimler ve Kurulum**

Proje, aşağıdaki Python bağımlılıklarına ihtiyaç duymaktadır:

| 🔧 Bileşen             | 🏷️ Sürüm |
| ---------------------- | -------- |
| **CUDA Sürümü**        | 11.8     |
| **Python Sürümü**      | >=3.8    |
| **PyTorch Sürümü**     | >=1.10.0 |
| **TorchVision Sürümü** | >=0.11.0 |
| **OpenCV Sürümü**      | >=4.5.3  |
| **Pandas**             | >=1.3.3  |
| **Matplotlib**         | >=3.4.3  |
| **Albumentations**     | >=1.4.15 |
| **ONNXRuntime Sürümü** | >=1.10.0 |
| **tqdm Sürümü**        | >=4.62.3 |

## **🐳 2. Docker Kullanımı**

Proje, Docker konteyneri içerisinde çalıştırılmak üzere yapılandırılmıştır.
Bu, test ortamında standartlaştırılmış bir çalışma ortamı sağlar.

### **📌 2.1. Docker İmajı Oluşturma**

Docker imajını oluşturmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
sudo docker build -f docker/Dockerfile -t kizilirmak_gorev-3.2.3 .
```

komut, mevcut Dockerfile kullanarak gerekli bağımlılıkları içeren bir Docker imajı oluşturacaktır.

### **📌 2.2. Docker Konteynerini Çalıştırma**

Oluşturulan imajı bir Docker konteyneri içinde çalıştırmak için:

```bash
sudo docker run --gpus all \
    -v /path/to/input/:/input \ # input olarak verilecek dosyanın yolu (Not: input olarak verilecek CSV dosyası ve video dosyaları ile aynı dizinde bulunmalıdır.)
    -v /path/to/output:/output \ # output.csv dosyasının kaydedileceği local (HOST) dizin
    kizilirmak_gorev-3.2.3
```

Burada:

- `/path/to/input/`: işlenecek video dosyaları ve input.csv dosyasının bulunduğu klasör
- `/path/to/output`: oluşturulacak çıktı dosyasının kaydedileceği klasör olmalıdır.

**Input.csv dosyası ve video dosyaları aynı input klasöründe bulunmalıdır!**

## **⚠️3. Docker Kullanımında Dikkat Edilmesi Gerekenler**

- Çıktı klasörü Docker içinde oluşturulacak ve **bağımlı volume olarak tanımlanmalıdır**.
- Çıktı dizinine Docker dışından erişim için izinleri düzenleyin:

```bash
sudo chown -R $USER:$USER /path/to/output/
```

## **📂 4. Klasör Yapısı**

```bash
│── 📂 proje_dizini
│   │── 📂 src
│   │   ├── algorithm.py
│   │   ├── model.pt
│   │── 📂 docker
│   │   ├── Dockerfile
│   │── 📂 dataset
│   │   ├── dataset.csv
│   │   ├── video_0001.png
│   │   ├── video_0002.png
│   │   ├── ...
│   │── README.md
```
