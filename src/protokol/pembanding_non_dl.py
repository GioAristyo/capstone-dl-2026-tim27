"""Pembanding Non-DL Tim 27: MFCC rata-rata dan simpangan + Random Forest / SVM.

PJ: Rosalia Siregar (123450036 - Analis Utama)

Cara menjalankan (pada data simulasi maupun data primer meja):
    python -m src.protokol.pembanding_non_dl --seed 2026
    python -m src.protokol.pembanding_non_dl --seed 2026 --algoritma svm
    python -m src.protokol.pembanding_non_dl --seed 2026 --final
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.fft import dct
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.common import evaluate as penilai  # noqa: E402
from src.protokol import data  # noqa: E402


def ekstrak_mfcc_rata_dan_simpangan(X_logmel: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """Ekstrak rata-rata dan simpangan baku MFCC dari spektrogram Log-mel (N, 64, 101).

    MFCC diperoleh melalui Discrete Cosine Transform (DCT-II) atas sumbu 64 pita Log-mel,
    kemudian diringkas menjadi mean dan std sepanjang sumbu waktu (101 frame).
    """
    # X_logmel: (N, 64, 101) -> DCT sepanjang axis=1 (frekuensi mel), ambil n_mfcc pertama
    mfcc = dct(X_logmel, type=2, axis=1, norm="ortho")[:, :n_mfcc, :]  # (N, 13, 101)

    mfcc_mean = mfcc.mean(axis=2)  # (N, 13)
    mfcc_std = mfcc.std(axis=2)    # (N, 13)

    # Tambahkan juga rata-rata dan simpangan energi total spektrogram per frame
    energi_frame = X_logmel.mean(axis=1)  # (N, 101)
    energi_mean = energi_frame.mean(axis=1, keepdims=True)  # (N, 1)
    energi_std = energi_frame.std(axis=1, keepdims=True)    # (N, 1)

    fitur = np.hstack([mfcc_mean, mfcc_std, energi_mean, energi_std]).astype(np.float32)  # (N, 28)
    return fitur


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--algoritma", choices=["rf", "svm", "otomatis"], default="otomatis",
                    help="rf: Random Forest, svm: SVM, otomatis: bandingkan RF & SVM pada validasi")
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    # 1. Muat data dengan pemisahan resmi dari src/protokol/data.py
    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    ytr = np.asarray(ytr, dtype=np.int64)
    yva = np.asarray(yva, dtype=np.int64)

    # 2. Ekstrak fitur MFCC rata-rata dan simpangan baku
    Ftr = ekstrak_mfcc_rata_dan_simpangan(Xtr)
    Fva = ekstrak_mfcc_rata_dan_simpangan(Xva)
    Fte = ekstrak_mfcc_rata_dan_simpangan(Xte) if (a.final and Xte is not None) else None

    # Normalisasi fitur statistik MFCC dihitung dari data latih saja
    scaler = StandardScaler()
    Ftr = scaler.fit_transform(Ftr)
    Fva = scaler.transform(Fva)
    if Fte is not None:
        Fte = scaler.transform(Fte)

    # 3. Latih Random Forest dan/atau SVM
    kandidat = {
        "rf": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=a.seed),
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=a.seed),
    }

    hasil_kandidat: dict[str, tuple[float, np.ndarray, object]] = {}
    for nama, clf in kandidat.items():
        clf.fit(Ftr, ytr)
        pred_v = clf.predict(Fva).astype(np.int64)
        skor_f1 = penilai.macro_f1(yva, pred_v)
        skor_acc = penilai.akurasi(yva, pred_v)
        hasil_kandidat[nama] = (skor_f1, pred_v, clf)
        print(f"  [Pembanding Non-DL {nama.upper()}] val_acc={skor_acc:.4f} | val_macro_f1={skor_f1:.4f}")

    if a.algoritma == "otomatis":
        terpilih = max(hasil_kandidat.keys(), key=lambda k: hasil_kandidat[k][0])
    else:
        terpilih = a.algoritma

    f1_terbaik, pred_val, model_terpilih = hasil_kandidat[terpilih]
    pred_test = model_terpilih.predict(Fte).astype(np.int64) if Fte is not None else None
    durasi = time.time() - t0

    # 4. Tulis prediksi dan panggil penilai resmi src.common.evaluate -> results/baseline/metrics.json
    out = AKAR / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", "baseline",
        "--model", "baseline",
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)

    metrics_json = json.loads((out / "metrics.json").read_text("utf-8"))
    print(f"\n[SELESAI] Pembanding Non-DL terpilih: {terpilih.upper()} (val_macro_f1={f1_terbaik:.4f})")
    print(f"Berkas valid: results/baseline/metrics.json -> {json.dumps(metrics_json['metrik'])}")


if __name__ == "__main__":
    main()
