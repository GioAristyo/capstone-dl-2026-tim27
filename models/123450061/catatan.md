
# Catatan model 123450061

## Alasan optimizer

Optimizer yang akan digunakan adalah Adam.
Pemilihan Adam dipertimbangkan karena optimizer ini
menggunakan adaptasi learning rate berdasarkan
estimasi momen gradien.

Konfigurasi learning rate dan optimizer akan
ditetapkan melalui eksperimen.

## Regularisasi

Regularisasi yang akan dievaluasi:
- Weight decay (L2).
- Dropout pada bagian representasi laten.

Pengaruh regularisasi akan diamati terhadap
reconstruction loss dan performa klasifikasi.

## Pencarian hyperparameter (tabel)

| Parameter | Kandidat |
|---|---|
| Latent dimension | 16, 32, 64 |
| Learning rate | 0.001, 0.0001 |
| Batch size | 32, 64, 128 |
| Weight decay | 0, 0.0001 |
| Epoch | Ditentukan melalui eksperimen |

## Ablation (>= 2)

1. AE dengan representasi laten tanpa dropout.
2. AE dengan representasi laten menggunakan dropout.

Perbandingan dilakukan untuk mengetahui pengaruh
regularisasi terhadap hasil model.

## Hasil 3 seed konfigurasi akhir

| Seed | Hasil | Keterangan |
|---|---|---|
| 1 | Belum diuji | Menunggu integrasi |
| 2 | Belum diuji | Menunggu integrasi |
| 3 | Belum diuji | Menunggu integrasi |

Hasil akhir akan diisi setelah eksperimen selesai.
