# Modul 1 – Dasar Deep Learning & Computer Vision

## Struktur Folder

```
P1/
├── data/
│   └── caltech101/        ← Letakkan dataset Caltech-101 di sini
│       ├── accordion/
│       ├── airplanes/
│       └── ...            (101 folder kelas)
├── models/
│   ├── mlp.py             ← Definisi SimpleMLP (dari P1)
│   └── alexnet.py         ← Definisi AlexNetCaltech (dari P4)
├── scripts/
│   ├── train_mlp.py       ← Training MLP pada data simulasi (dari P1)
│   └── train_alexnet_caltech.py  ← Training AlexNet pada Caltech-101 (dari P4)
├── utils/
│   ├── datasets.py        ← CustomImageDataset (dari P2)
│   └── training_loops.py  ← Fungsi evaluate (dari P2)
├── mnist_app.py           ← Aplikasi MNIST + GUI pygame (dari P3, standalone)
└── README.md
```

## Cara Menjalankan

### 1. Training MLP (data simulasi)
```bash
python scripts/train_mlp.py
```

### 2. Training AlexNet pada Caltech-101
Pastikan dataset sudah ada di `data/caltech101/` dengan struktur:
```
data/caltech101/
    accordion/
        image_0001.jpg
        ...
    airplanes/
        ...
```
Kemudian jalankan:
```bash
python scripts/train_alexnet_caltech.py
```

### 3. Aplikasi Klasifikasi Digit MNIST (GUI)
```bash
python mnist_app.py
```
Jika file `mnist_lenet.pth` belum ada, model akan dilatih otomatis terlebih dahulu.

## Catatan Dataset Caltech-101
- Letakkan folder-folder kelas langsung di dalam `data/caltech101/`
- Setiap subfolder = satu kelas, berisi file `.jpg` / `.jpeg` / `.png`
- Jumlah kelas akan otomatis terdeteksi dari jumlah subfolder


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

untuk yang resnet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from models.alexnet import AlexNetCaltech
from utils.datasets import CustomImageDataset
from utils.training_loops import evaluate