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
JALUR = "trf"


class AudioTransformer(nn.Module):
    """Encoder Transformer atas urutan bingkai waktu spektrogram Log-mel (101 langkah x 64 pita)."""

    def __init__(self, n_mels: int = 64, seq_len: int = 101, d_model: int = 64, nhead: int = 4, num_classes: int = 3):
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 101) -> ubah ke urutan waktu (B, 101, 64)
        x = x.transpose(1, 2)
        h = self.proj(x) + self.pos_emb[:, : x.size(1), :]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))  # Mean pooling sepanjang waktu
        return self.head(h)


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    Xtr_t = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    ytr_t = torch.tensor(np.asarray(ytr, dtype=np.int64))
    Xva_t = torch.tensor(np.asarray(Xva, dtype=np.float32))
    Xte_t = torch.tensor(np.asarray(Xte, dtype=np.float32)) if (final and Xte is not None) else None

    model = AudioTransformer(n_mels=Xtr_t.shape[1], seq_len=Xtr_t.shape[2], num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
