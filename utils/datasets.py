import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Struktur folder yang diharapkan:
        root_dir/
            kelas_a/
                gambar001.jpg
                ...
            kelas_b/
                gambarXYZ.jpg
        """
        self.root_dir = root_dir
        self.transform = transform

        # Ambil semua nama folder sebagai kelas
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Mapping nama kelas ke indeks
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.image_paths = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                # Ambil hanya file gambar
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_folder, fname)
                    self.image_paths.append((path, self.class_to_idx[cls]))

    def __len__(self):
        # Total jumlah gambar
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Ambil satu gambar berdasarkan indeks
        path, label = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
