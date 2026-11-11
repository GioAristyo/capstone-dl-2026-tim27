# Protokol Evaluasi Tim — Tim 27 (Taksiran Jarak Penutur dari Suara)

Diisi Penjaga Data & Evaluasi (Nabyla Sharfina - `123450008`), digabung ke `main` lewat *pull request* yang
disetujui pembimbing (Ardika Satria), lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Taksiran jarak penutur (1, 2, atau 3 meter) dari suara mikrofon meja |
| Sumber dataset | Data primer Meja Uji Ruang Cerdas 5 V (`t27-jarak-penutur.yaml`, 225 kondisi) & sesi simulasi |
| Jenis tugas (`TUGAS`) | `klasifikasi` (3 kelas jarak) |
| Target / label | Kolom `jarak` pada `label.csv`: `0` (`1 m`), `1` (`2 m`), `2` (`3 m`) |
| Ukuran data latih / validasi / uji | 135 kondisi (60%) / 45 kondisi (20%) / 45 kondisi (20%) — total 225 kondisi |
| Cara split | Pemisahan per grup partisipan (*participant-wise split*): Latih (`P01, P02, P03`), Validasi (`P04`), Uji (`P05`). Diperiksa dengan `assert` di `data.py` agar 0 irisan partisipan |
| Metrik utama untuk komparasi | `macro_f1` (F1-makro; rata-rata ± simpangan dari 3 seed) |
| Praproses yang sama untuk semua model | 1) Buang 1 detik pertama setiap kondisi; 2) Ambil 1 jendela 1 detik di sekitar ucapan (detik 1.0–2.0 pada 16 kHz); 3) Ekstraksi Log-mel 64 pita (`n_fft=512, hop_length=160`, bentuk `64x101`); 4) Normalisasi z-score (`mean` dan `std`) dihitung dari data latih saja |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop (`cpu laptop`) |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`), ditambah ukuran ONNX (KB) dan latensi (ms) di laptop meja |
| `data_hash` saat dibekukan | Dihitung otomatis oleh `src.common.manifest` dari `data/MANIFEST.json` (`split:2026`) |

Disetujui pembimbing pada: PR Protokol Tim 27
