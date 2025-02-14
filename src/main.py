import os
import csv
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

###########################################
# 1. Videodan Kare Çıkarma Fonksiyonu
###########################################
def extract_frames(video_path, num_frames=5):
    """
    Belirtilen video dosyasından, videonun uzunluğuna bağlı olarak num_frames adet kare çıkarır.
    
    Args:
        video_path (str): Video dosyasının yolu.
        num_frames (int): Videodan çıkarılacak kare sayısı.
    
    Returns:
        frames (list): Çıkarılan karelerin (numpy array) listesi.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []
    
    step = max(total_frames // num_frames, 1)
    frames = []
    frame_ids = [i for i in range(0, total_frames, step)]
    frame_ids = frame_ids[:num_frames]
    
    current_frame = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in frame_ids:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        current_frame += 1

    cap.release()
    return frames

###########################################
# 2. PyTorch Dataset Sınıfı (VideoDataset)
###########################################
class VideoDataset(Dataset):
    def __init__(self, csv_file, video_dir, transform=None, num_frames=5):
        """
        Test için video dosyası isimlerini içeren CSV dosyasını okuyup, 
        ilgili videoları belirtilen klasörden yükler ve her videodan belirli sayıda kare çıkarır.
        
        Args:
            csv_file (str): Test CSV dosyasının yolu. Dosyada sadece 'file_name' sütunu bulunmalıdır.
            video_dir (str): Videoların bulunduğu klasörün yolu.
            transform (callable, optional): Her kareye uygulanacak dönüşümler.
            num_frames (int): Her videodan çıkarılacak kare sayısı.
        """
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_filename = self.data.iloc[idx]['file_name']
        video_path = os.path.join(self.video_dir, video_filename)
        frames = extract_frames(video_path, num_frames=self.num_frames)
        
        if len(frames) == 0:
            raise ValueError(f"Video {video_filename} okunamadı veya kare bulunamadı.")
        
        processed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = cv2.resize(frame, (224, 224))
                frame = frame.transpose((2, 0, 1))
                frame = torch.tensor(frame, dtype=torch.float32) / 255.0
            processed_frames.append(frame)
        
        video_tensor = torch.stack(processed_frames)
        return video_tensor, video_filename

###########################################
# 3. Model Tanımı: FrameClassifier (ResNet18 ile)
###########################################
class FrameClassifier(nn.Module):
    def __init__(self, pretrained=False):
        """
        FrameClassifier, her bir kareyi işleyip binary (sahte/gerçek) tahmin yapabilen ResNet18 tabanlı bir modeldir.
        
        Args:
            pretrained (bool): Eğer önceden eğitilmiş ağırlıklar kullanılacaksa True, aksi halde False.
        """
        super(FrameClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)
        
    def forward(self, x):
        """
        Modelin ileri geçiş fonksiyonu.
        Args:
            x (tensor): [batch_size, 3, H, W] boyutunda giriş kareleri.
        Returns:
            tensor: [batch_size, 1] boyutunda çıkış logitleri.
        """
        return self.resnet(x)

###########################################
# 4. Inference ve Sonuçların CSV Dosyasına Yazılması
###########################################
def inference_and_generate_csv(model, dataloader, device, output_csv='/output/output.csv', threshold=0.5):
    """
    Yüklenmiş model ile inference yaparak, her video için deepfake tahminini hesaplar 
    ve sonuçları belirtilen formatta CSV dosyasına yazar.
    
    Args:
        model (nn.Module): Yüklenmiş model.
        dataloader (DataLoader): Test dataset'i içeren DataLoader.
        device (torch.device): İşlemin yapılacağı cihaz (CPU/GPU).
        output_csv (str): Oluşturulacak CSV dosyasının yolu.
        threshold (float): Tahmin için eşik değeri (sigmoid sonrası olasılık).
    """
    model.to(device)
    model.eval()
    
    results = []
    with torch.no_grad():
        for video_tensor, video_filename in dataloader:
            batch_size, num_frames, C, H, W = video_tensor.shape
            video_tensor = video_tensor.to(device)
            frames = video_tensor.view(-1, C, H, W)
            outputs = model(frames)
            outputs = outputs.view(batch_size, num_frames, 1)
            video_outputs = outputs.mean(dim=1)
            probs = torch.sigmoid(video_outputs).squeeze(1)
            predictions = (probs > threshold).int().cpu().numpy()
            
            for fname, pred in zip(video_filename, predictions):
                results.append({'file_name': fname, 'Is_fake': int(pred)})
    
    # CSV dosyasını oluşturma: mount edilmiş çıkış dizinine yazıyoruz.
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['file_name', 'Is_fake']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"{output_csv} dosyası oluşturuldu.")

###########################################
# 5. Ana Süreç: Yüklenmiş Model ile Tahmin (Inference)
###########################################
if __name__ == '__main__':
    # Docker konteynerindeki cihaz ayarını belirliyoruz.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ResNet18 için uygun dönüşümleri tanımlıyoruz:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Docker run komutuyla mount edilmiş dizinler kullanılıyor:
    # Giriş dosyası: /input/input.csv
    # Video dosyalarının bulunduğu klasör: /input/dataset
    # Çıkış dosyası: /output/output.csv
    input_csv = '/input/input.csv'
    video_dir = '/input/dataset'
    output_csv = '/output/output.csv'
    
    # Test dataset ve DataLoader'ı oluşturuyoruz.
    test_dataset = VideoDataset(csv_file=input_csv, video_dir=video_dir, transform=transform, num_frames=5)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Kaydedilmiş model dosyasını yüklüyoruz.
    model_save_path = 'model.pth'
    model = FrameClassifier(pretrained=False)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model {model_save_path} dosyasından yüklendi.")
    
    # İnference yaparak sonuçları çıkış dizinine yazıyoruz.
    inference_and_generate_csv(model, test_loader, device, output_csv=output_csv, threshold=0.5)
