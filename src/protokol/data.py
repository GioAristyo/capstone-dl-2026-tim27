"""Pemuat data kasus Tim 27 — DIBEKUKAN setelah Sprint 0 dengan persetujuan pembimbing.

Aturan protokol Tim 27:
1. Buang 1 detik pertama setiap kondisi; ambil jendela 1 detik di sekitar ucapan (1.0s - 2.0s).
2. Ekstraksi fitur: Log-mel spektrogram 64 pita (16 kHz, n_fft=512, hop_length=160) -> (64, 101).
3. Data uji dipisah per partisipan: Latih (P01, P02, P03), Validasi (P04), Uji (P05).
4. Normalisasi (mean & std) dihitung dari data latih saja.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav
import torch
import torchaudio.transforms as T

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.common import manifest

TUGAS = "klasifikasi"
AKAR = manifest.AKAR
DATA_DIR = AKAR / "data"

MAP_LABEL = {"1 m": 0, "2 m": 1, "3 m": 2}
PARTISIPAN_LATIH = ("P01", "P02", "P03")
PARTISIPAN_VAL = ("P04",)
PARTISIPAN_UJI = ("P05",)


def potong_jendela_ucapan(
    audio_path: Path,
    sample_rate_target: int = 16000,
    buang_awal_detik: float = 1.0,
    durasi_jendela_detik: float = 1.0,
) -> np.ndarray:
    """Baca berkas WAV 16 kHz, buang 1 detik pertama, kembalikan jendela 1 detik ucapan."""
    sr, sinyal = wav.read(str(audio_path))
    if sinyal.ndim > 1:
        sinyal = sinyal.mean(axis=1)

    if sinyal.dtype == np.int16:
        sinyal = sinyal.astype(np.float32) / 32768.0
    else:
        sinyal = sinyal.astype(np.float32)

    idx_mulai = int(buang_awal_detik * sr)
    idx_selesai = idx_mulai + int(durasi_jendela_detik * sr)

    if len(sinyal) < idx_selesai:
        kurang = idx_selesai - len(sinyal)
        sinyal = np.concatenate([sinyal, np.zeros(kurang, dtype=np.float32)])

    return sinyal[idx_mulai:idx_selesai]


def muat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kembalikan (X_train, y_train, X_val, y_val, X_test, y_test) terstandarisasi."""
    label_path = DATA_DIR / "label.csv"
    if not label_path.exists():
        from .buat_data_simulasi import buat_simulasi
        buat_simulasi()

    # Daftar kondisi terganggu (bila ada di data/kondisi_dikeluarkan.txt)
    abaikan_path = DATA_DIR / "kondisi_dikeluarkan.txt"
    kondisi_dikeluarkan: set[int] = set()
    if abaikan_path.exists():
        for baris in abaikan_path.read_text("utf-8").splitlines():
            baris = baris.strip()
            if baris and not baris.startswith("#") and baris.isdigit():
                kondisi_dikeluarkan.add(int(baris))

    mel_tf = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=512,
        hop_length=160,
        n_mels=64,
    )

    fitur_list, label_list, part_list = [], [], []

    with label_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_kondisi = int(row["kondisi"])
            if id_kondisi in kondisi_dikeluarkan:
                continue

            rel_audio = row.get("berkas_audio") or f"audio/kondisi_{id_kondisi:03d}.wav"
            audio_path = DATA_DIR / rel_audio
            if not audio_path.exists():
                audio_path = DATA_DIR / "audio" / Path(rel_audio).name

            jendela_1s = potong_jendela_ucapan(audio_path)
            tensor_wav = torch.from_numpy(jendela_1s).unsqueeze(0)
            with torch.no_grad():
                mel = mel_tf(tensor_wav)
                log_mel = torch.log(mel + 1e-6).squeeze(0).numpy()  # (64, 101)

            fitur_list.append(log_mel.astype(np.float32))
            label_list.append(MAP_LABEL[row["jarak"].strip()])
            part_list.append(row["partisipan"].strip())

    X = np.stack(fitur_list, axis=0)  # (N, 64, 101)
    y = np.array(label_list, dtype=np.int64)
    partisipan = np.array(part_list)

    idx_tr = np.where(np.isin(partisipan, PARTISIPAN_LATIH))[0]
    idx_va = np.where(np.isin(partisipan, PARTISIPAN_VAL))[0]
    idx_te = np.where(np.isin(partisipan, PARTISIPAN_UJI))[0]

    # Verifikasi kode: Tidak boleh ada partisipan yang muncul di lebih dari satu lipatan
    set_tr = set(partisipan[idx_tr])
    set_va = set(partisipan[idx_va])
    set_te = set(partisipan[idx_te])
    assert set_tr.isdisjoint(set_va), f"Kebocoran partisipan latih vs val: {set_tr & set_va}"
    assert set_tr.isdisjoint(set_te), f"Kebocoran partisipan latih vs uji: {set_tr & set_te}"
    assert set_va.isdisjoint(set_te), f"Kebocoran partisipan val vs uji: {set_va & set_te}"

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]
    X_te, y_te = X[idx_te], y[idx_te]

    # Normalisasi dihitung dari data latih saja
    mu = float(X_tr.mean())
    sigma = float(X_tr.std()) + 1e-6

    X_tr = (X_tr - mu) / sigma
    X_va = (X_va - mu) / sigma
    X_te = (X_te - mu) / sigma

    return X_tr, y_tr, X_va, y_va, X_te, y_te


if __name__ == "__main__":
    Xtr, ytr, Xva, yva, Xte, yte = muat()
    print(f"Verifikasi muat() berhasil!")
    print(f"  Latih    : X={Xtr.shape}, y={ytr.shape} (Partisipan: {PARTISIPAN_LATIH})")
    print(f"  Validasi : X={Xva.shape}, y={yva.shape} (Partisipan: {PARTISIPAN_VAL})")
    print(f"  Uji      : X={Xte.shape}, y={yte.shape} (Partisipan: {PARTISIPAN_UJI})")
    print("  Pemeriksaan kebocoran grup: LULUS (0 irisan partisipan antar-lipatan)")
