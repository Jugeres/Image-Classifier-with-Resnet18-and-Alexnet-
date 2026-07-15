import torch


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluasi model pada data loader yang diberikan.
    Mengembalikan akurasi (float).
    """
    model.eval()  # Nonaktifkan dropout & batchnorm

    correct, total = 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)               # Prediksi
        _, preds = torch.max(logits, 1)  # Ambil kelas tertinggi

        correct += (preds == yb).sum().item()
        total += yb.size(0)

    acc = correct / total  # Hitung akurasi
    return acc
