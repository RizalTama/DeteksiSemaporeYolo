# Web App Semaphore (Flask)

Versi web dari `app.py` (desktop) — deteksi huruf semaphore realtime lewat browser
memakai MJPEG streaming. Logika deteksi (model, fusi YOLO+pose, voting stabilisasi)
dipakai ulang persis dari `app.py`, tidak ada perubahan logika atau retraining.
File di folder root (`app.py`, model, dll) tidak disentuh sama sekali.

## Instalasi

Asumsi virtual environment root project sudah ada (lihat `../readme.md` / `../run-windows.md`).

```bash
cd web-app
source ../venv/bin/activate      # Windows: ..\venv\Scripts\activate
pip install -r requirements.txt  # menginstal ulang requirement root + flask
```

## Menjalankan

```bash
python server.py
```

Buka browser ke `http://localhost:5050` (port 5000 dihindari karena di macOS
sering dipakai AirPlay Receiver).

## Cara pakai

1. Klik **Mulai Kamera** (izinkan akses webcam dari OS kalau diminta).
2. Berdiri menghadap kamera, peragakan huruf semaphore sesuai `../panduan.png`.
3. Tahan pose sampai bar kuning penuh — huruf masuk ke kalimat.
4. **Simpan Frame** = tombol `S` di versi desktop; setelah 4 frame tersimpan,
   otomatis dibuat grid 2×2 (`output/grid_final.jpg`), bisa diunduh lewat tombol
   **Unduh Grid Final**.
5. **Reset** = tombol `R` di versi desktop (reset kalimat + hitungan simpan).
6. Klik **Stop** untuk melepas webcam sebelum menutup tab (setara tombol `Q`).

### Gestur tambahan (spasi & reset)

Selain tombol di panel, ada 2 gestur yang dibaca langsung dari kamera (dideteksi
lewat MediaPipe Hands, berdampingan dengan 2 model YOLO yang sudah ada):

| Gestur | Cara memicu | Aksi |
|---|---|---|
| **Berdiri tegap** (kedua lengan turun, sama seperti jeda antar huruf) | Ditahan diam &plusmn;7-8 detik | Tambah spasi ke kalimat yang terakumulasi |
| **Dua jari** (telunjuk+tengah lurus, jari lain menekuk, mirip simbol victori/peace) | Terdeteksi sekali | Reset kalimat langsung |

Progres gestur spasi terlihat di panel **"Gestur Spasi"** (bar hijau + label
"menahan..."/"jeda..."). Setelah spasi berhasil ditambahkan, ada jeda &plusmn;2-3
detik ("jeda...") sebelum sistem siap menghitung gestur spasi berikutnya, supaya
diam lama tidak menambah banyak spasi beruntun. Gestur reset menampilkan overlay
singkat **"Reset!"** di atas panel Kalimat.

Kalimat yang terakumulasi ditulis ke `web-app/hasil.txt` (terpisah dari
`hasil.txt` punya versi desktop di root).

## Struktur

```
web-app/
├── server.py          # routes Flask (/, /video_feed, /status, /start, /stop, /save, /reset)
├── camera.py           # thread background: buka webcam, jalankan deteksi, simpan state
├── detector.py         # model + logika deteksi, di-port dari app.py
├── templates/index.html
├── static/app.js
└── output/             # frame & grid_final.jpg hasil tombol Simpan Frame
```

## Catatan

- Model dimuat dari folder root project (`../best2_int8_openvino_model`,
  `../yolov8n-pose_int8_openvino_model`, dst — fallback ke `.pt` kalau folder
  OpenVINO tidak ada) — tidak ada file model yang di-copy ke `web-app/`.
- Konstanta tuning (`CONF`, `WINDOW`, `MIN_VOTES`, `GAP_RESET`, dst) ada di
  `detector.py`, nilainya identik dengan `app.py`.
- Konstanta gestur spasi/reset (`SPASI_DURASI`, `SPASI_COOLDOWN`,
  `RESET_FLASH_DURASI`) ada di `camera.py`; threshold gestur "dua jari"
  (`FINGER_STRAIGHT_ANGLE`) ada di `detector.py`.
- Server Flask dijalankan dengan `threaded=True` supaya endpoint `/status`,
  `/save`, `/reset` tidak terblokir selagi `/video_feed` streaming.
- Kamera dipegang oleh satu thread background yang di-share ke semua klien
  yang membuka `/video_feed` — bukan membuka kamera baru per koneksi.
