from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "cnn"


class SpektrogramCNN(nn.Module):
    """2D-CNN pada spektrogram Log-mel (1, 64, 101) untuk klasifikasi 3 jarak."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    Xtr_t = torch.tensor(np.asarray(Xtr, dtype=np.float32)).unsqueeze(1)  # (N, 1, 64, 101)
    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.int64))
    Xva_t = torch.tensor(np.asarray(Xva, dtype=np.float32)).unsqueeze(1)
    Xte_t = torch.tensor(np.asarray(Xte, dtype=np.float32)).unsqueeze(1) if (final and Xte is not None) else None

    model = SpektrogramCNN(num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 16
    n = len(Xtr_t)
    model.train()
    for _ in range(15):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(Xva_t).argmax(1).numpy()
        pred_test = model(Xte_t).argmax(1).numpy() if Xte_t is not None else None
    return pred_val, pred_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
