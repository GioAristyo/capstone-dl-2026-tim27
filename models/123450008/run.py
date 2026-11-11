"""Jalur Transfer Learning (tl) — Nabyla Sharfina (123450008).

    python models/123450008/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"


def bangun_model_tl(num_classes: int = 3) -> nn.Module:
    """Bangun ResNet-18 yang diadaptasi ke spektrogram 1-kanal (1, 64, 101)."""
    try:
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    except Exception:
        backbone = tv_models.resnet18(weights=None)

    # Adaptasi conv1 dari 3 kanal RGB ke 1 kanal spektrogram
    w_lama = backbone.conv1.weight.data  # (64, 3, 7, 7)
    conv1_baru = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1_baru.weight.data = w_lama.mean(dim=1, keepdim=True)
    backbone.conv1 = conv1_baru

    # Bekukan layer awal (layer1 & layer2) sebagai ekstraktor fitur dasar
    for nama, param in backbone.named_parameters():
        if nama.startswith(("layer1", "layer2")):
            param.requires_grad = False

    # Ganti kepala klasifikasi akhir untuk 3 kelas jarak
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return backbone


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    Xtr_t = torch.tensor(np.asarray(Xtr, dtype=np.float32)).unsqueeze(1)  # (N, 1, 64, 101)
    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.int64))
    Xva_t = torch.tensor(np.asarray(Xva, dtype=np.float32)).unsqueeze(1)
    Xte_t = torch.tensor(np.asarray(Xte, dtype=np.float32)).unsqueeze(1) if (final and Xte is not None) else None

    model = bangun_model_tl(num_classes=3)
    params_latih = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params_latih, lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 16
    n = len(Xtr_t)
    model.train()
    for _ in range(10):
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
