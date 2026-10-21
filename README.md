# 🚩 Deteksi Semaphore Realtime (YOLOv8 + Pose Estimation)

Aplikasi computer vision yang membaca huruf **semaphore (A–Z)** secara **realtime** dari webcam, lalu otomatis merangkainya menjadi kalimat. Tersedia dalam dua bentuk: aplikasi **desktop** (OpenCV) dan **web app** (Flask + MJPEG streaming).

Proyek ini menggabungkan dua pendekatan deteksi sekaligus agar hasilnya lebih akurat dan stabil dibanding memakai satu model saja:

| Model | Peran |
|---|---|
| 🟢 **best2** (YOLOv8, custom-trained) | Mengenali pose bendera semaphore secara visual langsung dari gambar |
| 🔵 **YOLOv8n-pose** | Membaca sudut kedua lengan (keypoints) lalu menerjemahkannya ke huruf sesuai tabel semaphore baku |

Hasil kedua model **difusi** — pose estimation diprioritaskan saat valid, YOLO mengisi sisanya — sehingga kombinasinya terbukti lebih akurat daripada masing-masing model sendirian.

---

## 📊 Performa Model

Hasil training `best2.pt` (YOLOv8n, 30 epoch, Ultralytics 8.3.235, GPU NVIDIA T4):

| Metrik | Nilai |
|---|---|
| **Precision** | 0.9513 |
| **Recall** | 0.9534 |
| **mAP50** | 0.9719 |
| **mAP50-95** | 0.6969 |

Diuji terhadap fusi YOLO + pose, hasilnya terbukti lebih stabil daripada tiap model berjalan sendiri (lihat `laporan.md` untuk rincian metodologi & metrik lengkap).

---

## ✨ Fitur

- 🎥 Deteksi realtime dari webcam, memakai dua model YOLO paralel (multi-thread)
- 🔤 Voting/stabilisasi antar-frame — huruf hanya diterima setelah pose ditahan stabil, tidak "nyeplos" karena noise satu frame
- 📝 Kalimat hasil deteksi otomatis tersimpan ke `hasil.txt`
- 🌐 Versi **web app** (Flask) dengan kontrol dari browser + gestur tangan tambahan (MediaPipe Hands) untuk spasi & reset kalimat
- 🖼️ Simpan cuplikan deteksi dan gabungkan otomatis jadi grid 2×2
- ⚡ Dukungan model OpenVINO INT8 untuk inferensi lebih cepat (fallback otomatis ke `.pt` jika tidak tersedia)
- 🖥️ Lintas platform: Windows, macOS, Linux

---

## 📁 Struktur Proyek

```
semaphore/
├── app.py                  # Aplikasi desktop (OpenCV)
├── web-app/                 # Aplikasi web (Flask + MJPEG streaming)
│   ├── server.py
│   ├── camera.py
│   ├── detector.py
│   ├── templates/ & static/
├── best2.pt / best2.onnx    # Model custom hasil training (deteksi pose bendera)
├── yolov8n-pose.pt          # Model pose estimation (Ultralytics)
├── best2_int8_openvino_model/        # Versi cepat (INT8) dari best2
├── yolov8n-pose_int8_openvino_model/ # Versi cepat (INT8) dari yolov8n-pose
├── dataset_training/ & dataset_validasi/  # Dataset training & validasi
├── panduan.png              # Panduan visual huruf semaphore A–Z
├── laporan.md                # Laporan lengkap hasil training & evaluasi model
└── requirements.txt
```

---

## 🚀 Instalasi

### Windows

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

> Pastikan Python **3.10–3.12** sudah terpasang dan **"Add Python to PATH"** dicentang saat instalasi. Panduan lebih detail (langkah demi langkah dari awal) ada di [`run-windows.md`](run-windows.md).

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

> Folder `best2_int8_openvino_model/` dan `yolov8n-pose_int8_openvino_model/` adalah versi cepat (INT8) dari model. Jika tidak tersedia, aplikasi otomatis memakai `best2.pt` / `yolov8n-pose.pt` (lebih lambat, tapi tetap jalan).

### Menjalankan versi web

```bash
cd web-app
pip install -r requirements.txt
python server.py
```

Lalu buka `http://localhost:5050` di browser. Detail lengkap ada di [`web-app/README.md`](web-app/README.md).

---

## ▶️ Cara Pakai

1. Berdiri menghadap kamera, peragakan huruf semaphore sesuai panduan di [`panduan.png`](panduan.png).
2. **Tahan pose** sampai bar kuning di layar penuh — huruf otomatis masuk ke kalimat.
3. Untuk huruf ganda (misal `"LL"`), **turunkan bendera sebentar**, lalu peragakan lagi.

| Tombol | Fungsi |
|:---:|---|
| `S` | Simpan gambar ke folder `output/` (setiap 4 gambar otomatis digabung jadi grid 2×2) |
| `R` | Reset kalimat |
| `Q` | Keluar |

Kalimat hasil deteksi otomatis tersimpan di `hasil.txt`.

---

## ⚙️ Konfigurasi

Beberapa konstanta penting di bagian atas `app.py`:

| Konstanta | Fungsi |
|---|---|
| `MIN_VOTES` / `WINDOW` | Trade-off kecepatan vs kestabilan penangkapan huruf (turunkan ke `6`/`8` jika terasa lambat, naikkan jika banyak huruf salah terbaca) |
| `GAP_RESET` | Berapa lama bendera harus diturunkan sebelum huruf yang sama boleh diulang |
| `CONF` | Ambang confidence deteksi YOLO |

---

## 🛠️ Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — deteksi objek & pose estimation
- OpenCV — capture & render video
- OpenVINO — inferensi INT8 yang dioptimalkan
- Flask + MediaPipe Hands — versi web dan deteksi gestur tangan
- NumPy, PyTorch, ONNX

---

## 📄 Dokumentasi Lainnya

- [`laporan.md`](laporan.md) — laporan lengkap hasil training & evaluasi model
- [`run-windows.md`](run-windows.md) — panduan instalasi Windows dari nol
- [`web-app/README.md`](web-app/README.md) — dokumentasi lengkap versi web
