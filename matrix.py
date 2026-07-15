import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.alexnet import AlexNetCaltech
from utils.datasets import CustomImageDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/caltech101"
BATCH_SIZE = 32

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = CustomImageDataset(root_dir=DATA_DIR,
                                  transform=data_transforms)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

model = AlexNetCaltech(num_classes=len(full_dataset.classes)).to(DEVICE)
all_preds = []
all_labels = []

model.load_state_dict(torch.load("alexnet_caltech_normal.pth"))
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)


plt.figure(figsize=(12, 10))

sns.heatmap(cm, cmap='Blues', annot=False)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.tight_layout()  
plt.show()