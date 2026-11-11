
"""Kerangka model Autoencoder (AE) - Tim 27.

    python models/123450061/run.py --seed 1
    python models/123450061/run.py --seed 1 --final
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "ae"


def latih_dan_prediksi(
    seed: int,
    final: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Melatih Autoencoder dan menghasilkan prediksi validasi/uji."""

    import copy
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # ==================================================
    # 1. REPRODUKSIBILITAS
    # ==================================================

    torch.manual_seed(seed)
    np.random.seed(seed)

    perangkat = torch.device("cpu")

    # ==================================================
    # 2. MEMUAT DATA SESUAI PROTOKOL TIM
    # ==================================================

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    # Memastikan bentuk data Log-Mel sesuai protokol
    assert Xtr.shape[1:] == (64, 101), (
        f"Bentuk data tidak sesuai: {Xtr.shape}"
    )

    # Mengubah data menjadi tensor
    def ke_tensor(X):
        return torch.tensor(
            np.asarray(X, dtype=np.float32),
            dtype=torch.float32,
            device=perangkat
        ).reshape(len(X), -1)

    X_train = ke_tensor(Xtr)
    X_val = ke_tensor(Xva)

    y_train = torch.tensor(
        np.asarray(ytr, dtype=np.int64),
        dtype=torch.long,
        device=perangkat
    )

    y_val = torch.tensor(
        np.asarray(yva, dtype=np.int64),
        dtype=torch.long,
        device=perangkat
    )

    # ==================================================
    # 3. MEMBUAT ARSITEKTUR AUTOENCODER
    # ==================================================

    INPUT_DIM = 64 * 101
    LATENT_DIM = 128

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(INPUT_DIM, 512),
                nn.ReLU(),
                nn.Linear(512, LATENT_DIM)
            )

            self.decoder = nn.Sequential(
                nn.Linear(LATENT_DIM, 512),
                nn.ReLU(),
                nn.Linear(512, INPUT_DIM)
            )

        def forward(self, x):
            z = self.encoder(x)
            reconstruction = self.decoder(z)
            return reconstruction

        def encode(self, x):
            return self.encoder(x)

    model = Autoencoder().to(perangkat)

    # ==================================================
    # 4. PELATIHAN AUTOENCODER
    # ==================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    criterion = nn.MSELoss()

    dataset_train = TensorDataset(X_train)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset_train,
        batch_size=16,
        shuffle=True,
        generator=generator
    )

    EPOCH_AE = 100
    PATIENCE_AE = 15

    best_val_loss = float("inf")
    best_model = None
    counter = 0

    for epoch in range(EPOCH_AE):

        model.train()
        total_loss = 0.0

        for (batch,) in loader:

            optimizer.zero_grad()

            reconstruction = model(batch)

            loss = criterion(reconstruction, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        train_loss = total_loss / len(X_train)

        # Evaluasi rekonstruksi pada data validasi
        model.eval()

        with torch.no_grad():
            reconstruction_val = model(X_val)
            val_loss = criterion(
                reconstruction_val,
                X_val
            ).item()

        # Early stopping berdasarkan reconstruction loss
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            counter = 0

        else:
            counter += 1

        if counter >= PATIENCE_AE:
            break

    # Menggunakan bobot AE terbaik
    if best_model is None:
        raise RuntimeError("Pelatihan Autoencoder gagal.")

    model.load_state_dict(best_model)

    print(
        f"[AE] Epoch: {epoch + 1}, "
        f"Train Loss: {train_loss:.6f}, "
        f"Val Loss: {best_val_loss:.6f}"
    )

    # ==================================================
    # 5. EKSTRAKSI REPRESENTASI LATEN
    # ==================================================

    model.eval()

    with torch.no_grad():

        Z_train = model.encode(X_train)
        Z_val = model.encode(X_val)

    # ==================================================
    # 6. PENAKSIR SEDERHANA / LINEAR CLASSIFIER
    # ==================================================

    classifier = nn.Linear(
        LATENT_DIM,
        3
    ).to(perangkat)

    optimizer_cls = torch.optim.Adam(
        classifier.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    criterion_cls = nn.CrossEntropyLoss()

    EPOCH_CLS = 200
    PATIENCE_CLS = 20

    best_acc = -1.0
    best_classifier = None
    counter_cls = 0

    for epoch in range(EPOCH_CLS):

        classifier.train()

        optimizer_cls.zero_grad()

        output = classifier(Z_train)

        loss = criterion_cls(
            output,
            y_train
        )

        loss.backward()
        optimizer_cls.step()

        # Evaluasi classifier menggunakan validasi
        classifier.eval()

        with torch.no_grad():

            output_val = classifier(Z_val)

            pred_val_tensor = output_val.argmax(dim=1)

            val_acc = (
                pred_val_tensor == y_val
            ).float().mean().item()

        if val_acc > best_acc:

            best_acc = val_acc
            best_classifier = copy.deepcopy(
                classifier.state_dict()
            )
            counter_cls = 0

        else:
            counter_cls += 1

        if counter_cls >= PATIENCE_CLS:
            break

    if best_classifier is None:
        raise RuntimeError("Pelatihan classifier gagal.")

    classifier.load_state_dict(best_classifier)
    classifier.eval()

    # ==================================================
    # 7. PREDIKSI DATA VALIDASI
    # ==================================================

    with torch.no_grad():

        pred_val = classifier(Z_val).argmax(dim=1)

        pred_val = pred_val.cpu().numpy().astype(np.int64)

    # ==================================================
    # 8. PREDIKSI DATA UJI (HANYA JIKA --final)
    # ==================================================

    pred_test = None

    if final:

        X_test = ke_tensor(Xte)

        with torch.no_grad():

            Z_test = model.encode(X_test)

            pred_test = classifier(
                Z_test
            ).argmax(dim=1)

            pred_test = (
                pred_test.cpu()
                .numpy()
                .astype(np.int64)
            )

    return pred_val, pred_test


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--seed",
        type=int,
        required=True
    )

    ap.add_argument(
        "--final",
        action="store_true",
        help="Sprint terakhir: evaluasi data uji sekali"
    )

    ap.add_argument(
        "--perangkat",
        default="cpu laptop"
    )

    a = ap.parse_args()

    t0 = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        a.seed,
        a.final
    )

    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    np.save(
        out / "pred_val.npy",
        np.asarray(pred_val)
    )

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat
    ]

    if pred_test is not None:

        np.save(
            out / "pred_test.npy",
            np.asarray(pred_test)
        )

        perintah += [
            "--pred-test",
            str(out / "pred_test.npy")
        ]

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR
    )


if __name__ == "__main__":
    main()
