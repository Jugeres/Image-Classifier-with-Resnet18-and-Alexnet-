import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from models.alexnet import AlexNetCaltech
from utils.datasets import CustomImageDataset
from utils.training_loops import evaluate

# Gunakan GPU jika tersedia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path dataset Caltech-101 — letakkan dataset Anda di folder ini
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'caltech101')
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10

# Transformasi gambar
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Muat dataset Caltech-101 dari folder data/caltech101/
full_dataset = CustomImageDataset(
    root_dir=DATA_DIR,
    transform=data_transforms
)

print(f"Total gambar: {len(full_dataset)}")
print(f"Jumlah kelas: {len(full_dataset.classes)}")

# Split dataset (80% train, 20% validasi)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_set, val_set = random_split(
    full_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Inisialisasi model dengan jumlah kelas sesuai dataset
model = AlexNetCaltech(
    num_classes=len(full_dataset.classes)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



def evaluate_with_loss(model, loader, criterion, device):
    """Evaluasi model: kembalikan loss rata-rata dan akurasi."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Training
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Reset gradien
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Hitung loss
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Update bobot
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Hitung akurasi training
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / total_train
    train_acc = correct_train / total_train

    # Validasi: loss + akurasi
    val_loss, val_acc = evaluate_with_loss(model, val_loader, criterion, DEVICE)
    scheduler.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# Simpan model setelah selesai training
torch.save(model.state_dict(), "alexnet_caltech.pth")
print("Model tersimpan sebagai alexnet_caltech.pth")