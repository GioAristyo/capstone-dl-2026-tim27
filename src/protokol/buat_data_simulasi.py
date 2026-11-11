
"""Simulator Rencana Eksperimen Meja Tim 27 (Pengganti meja.py --simulasi).

Membangkitkan berkas simulasi untuk 225 kondisi:
15 blok x 15 kondisi = 225 kondisi.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav

AKAR = Path(__file__).resolve().parents[2]
DATA_DIR = AKAR / "data"
AUDIO_DIR = DATA_DIR / "audio"


def buat_simulasi(seed: int = 1027) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    jarak_list = ["1 m", "2 m", "3 m"]
    partisipan_list = ["P01", "P02", "P03", "P04", "P05"]
    kata_list = ["nyalakan", "matikan", "kipas", "lampu", "berhenti"]

    ulangan = 3
    durasi_kondisi = 2.5
    jeda_detik = 0.5
    sample_rate = 16000
    n_samples = int(durasi_kondisi * sample_rate)

    # Membentuk dan mengacak 15 blok
    blok_kombinasi = [
        (j, p) for j in jarak_list for p in partisipan_list
    ]
    rng.shuffle(blok_kombinasi)

    total_blok = len(blok_kombinasi)
    kondisi_per_blok = len(kata_list) * ulangan
    total_kondisi = total_blok * kondisi_per_blok

    estimasi_menit = (
        total_kondisi * (durasi_kondisi + jeda_detik)
        + total_blok * 60
    ) / 60.0

    print("=== GLADI RENCANA MODE SIMULASI: TIM 27 ===")
    print(f"Rencana        : t27-jarak-penutur.yaml (seed={seed})")
    print(f"Jumlah blok    : {total_blok} blok (3 jarak x 5 partisipan)")
    print(
        f"Kondisi/blok   : {kondisi_per_blok} kondisi "
        f"(5 kata x {ulangan} ulangan)"
    )
    print(f"Total kondisi  : {total_kondisi} kondisi")
    print(f"Durasi kondisi : {durasi_kondisi} s (+ jeda {jeda_detik} s)")
    print(
        f"Perkiraan waktu: ~{estimasi_menit:.1f} menit "
        "(muat dalam slot 50 menit efektif)"
    )

    # Skala amplitudo berdasarkan jarak
    skala_energi = {
        "1 m": 0.70,
        "2 m": 0.35,
        "3 m": 0.12,
    }

    label_rows = [
        "kondisi,blok,partisipan,jarak,kata,ulangan,"
        "t_mulai,t_selesai,berkas_audio"
    ]
    index_rows = [
        "kondisi,berkas,sample_rate,durasi_detik"
    ]

    t_curr = 1762927200.0
    kondisi_id = 1

    for id_blok, (jarak, partisipan) in enumerate(
        blok_kombinasi, start=1
    ):
        kondisi_blok = [
            (k, u)
            for u in range(1, ulangan + 1)
            for k in kata_list
        ]
        rng.shuffle(kondisi_blok)

        for kata, u in kondisi_blok:
            t_mulai = t_curr
            t_selesai = t_curr + durasi_kondisi

            nama_wav = f"kondisi_{kondisi_id:03d}.wav"
            rel_audio = f"audio/{nama_wav}"

            # Audio simulasi berupa bising latar
            audio = rng.normal(
                0, 0.005, size=n_samples
            ).astype(np.float32)

            # Sinyal ucapan tiruan
            t_sp = np.linspace(
                0, 1.0, sample_rate,
                endpoint=False,
                dtype=np.float32,
            )

            f0 = (
                120.0
                + (abs(hash(partisipan)) % 70)
                + (abs(hash(kata)) % 30)
            )

            amp = skala_energi[jarak] * (
                0.9 + 0.2 * rng.random()
            )

            ucapan = (
                0.55 * np.sin(2 * np.pi * f0 * t_sp)
                + 0.30 * np.sin(2 * np.pi * 2 * f0 * t_sp)
                + 0.15 * np.sin(2 * np.pi * 3 * f0 * t_sp)
            ) * amp

            audio[16000:32000] += (
                ucapan * np.hanning(sample_rate)
            )

            audio_int16 = (
                np.clip(audio, -1.0, 1.0) * 32767
            ).astype(np.int16)

            wav.write(
                str(AUDIO_DIR / nama_wav),
                sample_rate,
                audio_int16,
            )

            label_rows.append(
                f"{kondisi_id},{id_blok},{partisipan},"
                f"{jarak},{kata},{u},"
                f"{t_mulai:.2f},{t_selesai:.2f},{rel_audio}"
            )

            index_rows.append(
                f"{kondisi_id},{rel_audio},"
                f"{sample_rate},{durasi_kondisi}"
            )

            kondisi_id += 1
            t_curr = t_selesai + jeda_detik

    (DATA_DIR / "label.csv").write_text(
        "\n".join(label_rows) + "\n",
        encoding="utf-8",
    )

    (DATA_DIR / "audio_index.csv").write_text(
        "\n".join(index_rows) + "\n",
        encoding="utf-8",
    )

    print(
        f"[SELESAI] {total_kondisi} kondisi tersimpan "
        "di data/label.csv dan data/audio/"
    )


if __name__ == "__main__":
    buat_simulasi()
