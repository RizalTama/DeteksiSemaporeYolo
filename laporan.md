# Laporan Hasil Training Model YOLO — Deteksi Semaphore A–Z

> Rangkuman untuk bahan penulisan BAB IV. Sumber data: metadata training yang
> tersimpan di dalam checkpoint `best2.pt` (training dilakukan di Google Colab,
> sehingga folder `runs/train/` beserta `results.csv`/`results.png` aslinya
> **tidak ada di project ini** — riwayat per-epoch berhasil diekstrak kembali
> dari checkpoint dan grafiknya dibuat ulang di `laporan_assets/`), ditambah
> hasil pengujian lokal terhadap 26 foto klien dan pengujian realtime.
> Bagian yang datanya tidak tersedia ditandai **[tidak tersedia]**.

---

## 1. Info Dataset

| Item | Nilai |
|---|---|
| Jumlah kelas | 26 (huruf semaphore A–Z) |
| Sumber dataset (training awal) | Dataset publik Roboflow — [**Semaphore Detection 60-20-20**](https://universe.roboflow.com/skripsi-cyber-university/semaphore-detection-60-20-20) (split train/valid/test 60/20/20 sesuai nama project). Halaman Roboflow tidak bisa diakses otomatis untuk ekstraksi jumlah citra pasti (server menolak fetch non-browser) — **jumlah total gambar & breakdown per huruf dari dataset ini perlu dicek manual di dashboard Roboflow** |
| Fine-tuning | Dilanjutkan dengan fine-tuning memakai dataset foto klien (`client-yolo/semaphore` di Google Drive akun training) — tidak ikut tersimpan di project ini |
| Total gambar training | **[tidak tersedia secara lokal]** — gabungan dataset Roboflow (awal) + dataset klien (fine-tuning), tidak tersimpan di project ini. Cek langsung di halaman browse Roboflow untuk angka pasti: https://universe.roboflow.com/skripsi-cyber-university/semaphore-detection-60-20-20/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true |
| Breakdown per huruf | **[tidak tersedia]** — alasan sama |
| Data uji lokal (tambahan) | 26 foto klien (1 foto/huruf, `Data_Client/`), diberi label dan dipakai sebagai set pengujian independen (`dataset_validasi/`) |

## 2. Konfigurasi Training

| Parameter | Nilai |
|---|---|
| Base model | **YOLOv8n** (pretrained `yolov8n.pt`, 3.010.718 parameter) |
| Framework | Ultralytics 8.3.235, training tanggal 2025-12-09 (Google Colab) |
| Epoch | **30** |
| Batch size | **16** |
| Image size (imgsz) | **832** |
| Optimizer | **AdamW** |
| Learning rate | Dari log per-epoch: warm-up ±0,067 → stabil di ±0,0005–0,001 setelah epoch 3 (skema auto Ultralytics) |
| Augmentasi | mosaic 0,7 (dimatikan 10 epoch terakhir / `close_mosaic=10`), degrees 25, translate 0,1, scale 0,5, hsv_h 0,015, hsv_s 0,7, hsv_v 0,4, **fliplr 0,0** (flip horizontal sengaja dimatikan karena pose semaphore bersifat kiral — dicerminkan berarti huruf berbeda), **flipud 0,0** (default Ultralytics, tidak diubah — flip vertikal juga akan mengubah identitas huruf karena posisi lengan "atas" vs "bawah" bisa tertukar) |
| Lain-lain | patience 100, seed 0, run name `hasil_fix` |
| Total waktu training | **±32,5 menit** (kolom waktu kumulatif log: 1.949 detik untuk 30 epoch, ±65 detik/epoch) |
| Perangkat training | GPU Google Colab — **NVIDIA T4** |

## 3. Hasil Metrik Evaluasi

Nilai **epoch terakhir (30)** pada set validasi — inilah bobot yang tersimpan sebagai `best2.pt`:

| Metrik | Nilai |
|---|---|
| **Precision** | **0,9513** |
| **Recall** | **0,9534** |
| **mAP50** | **0,9719** |
| **mAP50-95** | **0,6969** |
| val box loss | 1,0018 |
| val cls loss | 0,6390 |
| val dfl loss | 1,6548 |
| train box loss | 0,7912 |
| train cls loss | 0,6185 |
| train dfl loss | 1,5795 |

Nilai puncak selama training: mAP50 tertinggi **0,9786** (epoch 28); mAP50-95 tertinggi **0,6969** (epoch 30 — masih menanjak saat training berhenti, indikasi jumlah epoch dapat ditambah).

**Lokasi file grafik:**

- `results.png` / `confusion_matrix.png` **asli** dari training: berada di Google Drive Colab `/content/drive/MyDrive/client-yolo/results/hasil_fix/` — **tidak ada di project ini**.
- Grafik **rekonstruksi** (dibuat ulang dari riwayat per-epoch di dalam checkpoint, nilainya identik dengan aslinya):
  - `laporan_assets/kurva_loss.png` — kurva box/cls/dfl loss (train vs val), pengganti results.png
  - `laporan_assets/kurva_metrik.png` — kurva precision, recall, mAP50, mAP50-95 per epoch
  - `laporan_assets/riwayat_training.csv` — tabel angka mentah 30 epoch (pengganti results.csv)
- Confusion matrix (dari pengujian 26 foto klien, bukan dari set validasi training):
  - `laporan_assets/confusion_matrix_uji.png`

## 4. Hasil Pengujian Real-Time

**Perangkat uji:** MacBook Pro, CPU Intel Core i7-1068NG7 @ 2,30 GHz (4 core), tanpa GPU; webcam internal (resolusi default, tidak dicatat).

**Kecepatan inferensi** (deteksi huruf @832 px + estimasi pose @640 px per frame):

| Konfigurasi | Waktu/frame | FPS |
|---|---|---|
| PyTorch FP32, dua model berurutan | 192 ms | ±5,2 |
| OpenVINO INT8, berurutan | 78 ms | ±12,8 |
| **OpenVINO INT8, dua model paralel (dipakai aplikasi)** | **51 ms** | **±19,7** |

**Akurasi pengenalan huruf** (uji 26 foto klien, 1 foto/huruf, kondisi dalam ruangan, latar pintu/dinding, pencahayaan lampu ruangan):

| Sistem | Benar | Akurasi |
|---|---|---|
| YOLO `best2.pt` saja | 20/26 | 76,9% |
| Dekoder pose (sudut lengan) saja | 23/26 | 88,5% |
| **Hibrida YOLO + pose (sistem final)** | **24/26** | **92,3%** |

**Analisis kegagalan model** (bahan pembahasan):

1. **Kesenjangan domain (domain gap):** metrik pada set validasi training sangat tinggi (precision 95%) tetapi pada foto kondisi nyata akurasi YOLO turun ke 76,9% — kondisi kamera, pencahayaan, dan latar berbeda dari data training.
2. **Kebingungan sudut lengan tegak vs 45°:** kesalahan YOLO bersifat sistematis pada pasangan huruf yang hanya berbeda sudut satu lengan: Y→J, T→J, Q→P, L→V. Ini yang mendorong penambahan dekoder pose geometris.
3. **Bendera kedua tidak terdeteksi:** S terbaca B (huruf dua-bendera terbaca sebagai huruf satu-bendera).
4. **Huruf lengan-menyilang:** H terbaca O (baik oleh YOLO maupun dekoder pose) — posisi rendah vs tinggi pada lengan menyilang sulit dibedakan; butuh data tambahan.
5. **Pose peraga yang salah:** foto huruf T dari klien ternyata diperagakan tercermin (mirror) dari panduan (`panduan.png`) — kegagalan pada T bukan murni kesalahan model.
6. **Confidence rendah pada beberapa huruf benar** (D 0,15; B 0,18; W 0,30) — pada mode realtime menyebabkan deteksi berkedip; diatasi dengan mekanisme voting antar frame (huruf diterima bila mayoritas ≥9 dari 12 frame).
7. **Deteksi palsu saat orang tidak memperagakan:** orang duduk di depan laptop (tangan di meja) terbaca sebagai huruf oleh dekoder pose — diatasi dengan syarat seluruh badan (termasuk pinggul) terlihat di frame.

## 5. Struktur Folder Project

```
semaphore/
├── app.py                              # aplikasi realtime (YOLO + pose, voting)
├── best2.pt                            # model YOLO hasil training (utama)
├── best2_int8_openvino_model/          # model terkuantisasi INT8 (produksi)
├── best2_openvino_model/               # ekspor OpenVINO FP32 (perantara)
├── best2.onnx                          # ekspor ONNX (perantara)
├── yolov8n-pose.pt                     # model pose COCO (pendamping)
├── yolov8n-pose_int8_openvino_model/   # versi INT8-nya
├── yolov8n-pose_openvino_model/        # versi FP32-nya
├── yolov8n.pt                          # YOLOv8n COCO (dipakai saat pelabelan)
├── panduan.png                         # panduan pose semaphore (acuan resmi)
├── Data_Client/                        # 26 foto mentah klien (1 per huruf)
├── dataset_validasi/                   # foto klien tersusun per huruf A–Z + hasil_validasi.csv
├── dataset_training/                   # dataset format YOLO (images/, labels/, preview/, data.yaml)
├── laporan_assets/                     # grafik & tabel untuk laporan ini
├── output/                             # simpanan gambar dari aplikasi (kosong; terisi saat tombol S ditekan)
├── dist/semaphore_client.zip           # paket distribusi untuk klien
├── readme.md / run-windows.md          # panduan instalasi & penggunaan
└── requirements.txt
```

## 6. Lokasi File Pendukung (lampiran/gambar skripsi)

| Keperluan | Path |
|---|---|
| Confusion matrix (uji 26 foto klien) | `laporan_assets/confusion_matrix_uji.png` |
| Grafik loss training (box/cls/dfl, train vs val) | `laporan_assets/kurva_loss.png` |
| Grafik metrik (precision, recall, mAP50, mAP50-95) | `laporan_assets/kurva_metrik.png` |
| Data mentah riwayat 30 epoch | `laporan_assets/riwayat_training.csv` |
| Tabel prediksi vs label per foto uji | `dataset_validasi/hasil_validasi.csv` |
| Contoh gambar deteksi (kotak + label per huruf, 26 gambar) | `dataset_training/preview/A_1.jpg` … `Z_1.jpg` |
| Panduan pose semaphore | `panduan.png` |
| Foto uji mentah dari klien | `Data_Client/` (26 file .jpeg) |
| Screenshot aplikasi realtime | **[belum ada]** — folder `output/` masih kosong; jalankan aplikasi dan tekan **S** untuk menyimpan contoh deteksi live (4 gambar otomatis menjadi grid 2×2 di `output/grid_final.jpg`) |
| Dokumen legal (Turnitin, persetujuan publikasi) | **[di luar project ini]** — proses cek similarity/Turnitin ditangani terpisah oleh tim JokiCoding, bukan disimpan di repo ini |

---

*Catatan metodologi: seluruh angka pada Bagian 2–3 diekstrak langsung dari metadata yang tersimpan di dalam file `best2.pt` (train_args, train_metrics, dan riwayat train_results per epoch), sehingga identik dengan isi results.csv asli di Colab. Angka Bagian 4 diukur langsung di perangkat uji pada project ini (Juli 2026).*
