# Cara Menjalankan Web App (Flask)

Panduan menjalankan versi web dari aplikasi deteksi semaphore (`web-app/`).
Pilih bagian sesuai OS yang dipakai.

---

## Windows

1. Buka **Command Prompt** (atau PowerShell) di folder utama project (folder yang berisi `app.py`).
2. Kalau virtual environment (`venv`) belum pernah dibuat, buat dulu:
   ```bat
   python -m venv venv
   ```
3. Aktifkan virtual environment:
   ```bat
   venv\Scripts\activate
   ```
   Berhasil kalau muncul `(venv)` di awal baris.
4. Masuk ke folder `web-app`:
   ```bat
   cd web-app
   ```
5. Install dependency (sekali saja, kalau belum pernah):
   ```bat
   pip install -r requirements.txt
   ```
6. Jalankan server:
   ```bat
   python server.py
   ```
7. Buka browser, akses:
   ```
   http://localhost:5050
   ```
8. Klik **Mulai Kamera** di halaman web (izinkan akses webcam kalau Windows minta izin).
9. Selain deteksi huruf semaphore, ada 2 gestur tambahan (lihat bagian **Gestur tambahan** di bawah).
10. Untuk berhenti: tekan `Ctrl + C` di jendela Command Prompt.

**Kalau muncul error "port sedang dipakai"**: tutup dulu proses `python server.py` yang mungkin masih berjalan dari sesi sebelumnya, lalu ulangi langkah 6.

---

## macOS

1. Buka **Terminal** di folder utama project (folder yang berisi `app.py`).
2. Kalau virtual environment (`venv`) belum pernah dibuat, buat dulu:
   ```bash
   python3 -m venv venv
   ```
3. Aktifkan virtual environment:
   ```bash
   source venv/bin/activate
   ```
   Berhasil kalau muncul `(venv)` di awal baris.
4. Masuk ke folder `web-app`:
   ```bash
   cd web-app
   ```
5. Install dependency (sekali saja, kalau belum pernah):
   ```bash
   pip install -r requirements.txt
   ```
6. Jalankan server:
   ```bash
   python server.py
   ```
7. Buka browser, akses:
   ```
   http://localhost:5050
   ```
8. Klik **Mulai Kamera** di halaman web (izinkan akses webcam kalau macOS minta izin).
9. Selain deteksi huruf semaphore, ada 2 gestur tambahan (lihat bagian **Gestur tambahan** di bawah).
10. Untuk berhenti: tekan `Ctrl + C` di jendela Terminal.

**Kalau muncul error "Address already in use" / "port sedang dipakai"**: port 5050 kemungkinan masih dipakai instance `server.py` sebelumnya yang belum ditutup. Cari dan matikan prosesnya:
```bash
lsof -nP -iTCP:5050 -sTCP:LISTEN
kill <PID_yang_muncul>
```
lalu ulangi langkah 6.

---

## Gestur tambahan

Selain deteksi huruf semaphore, ada 2 gestur yang dibaca langsung dari kamera lewat
MediaPipe Hands, berdampingan dengan model YOLO yang sudah ada:

| Gestur | Cara memicu | Aksi |
|---|---|---|
| **Berdiri tegap** (kedua lengan turun, sama seperti jeda antar huruf) | Ditahan diam &plusmn;7-8 detik | Tambah spasi ke kalimat yang terakumulasi |
| **Dua jari** (telunjuk+tengah lurus, jari lain menekuk, mirip simbol victori/peace) | Terdeteksi sekali | Reset kalimat langsung |

Progres gestur spasi terlihat di panel **"Gestur Spasi"** di halaman web. Gestur
reset menampilkan overlay singkat di atas panel Kalimat.

---

## Catatan

- Port yang dipakai **5050**, bukan 5000 — karena di macOS port 5000 sering dipakai oleh AirPlay Receiver.
- File model (`best2*`, `yolov8n-pose*`) tidak perlu di-copy ke folder `web-app` — server otomatis mengambilnya dari folder utama project.
- Tombol di halaman web setara dengan shortcut keyboard versi desktop: **Simpan Frame** = `S`, **Reset** = `R`, **Stop** = `Q`.
- `requirements.txt` di `web-app` sekarang juga menginstal **mediapipe** (dipakai untuk gestur tambahan di atas) selain flask — instalasi langkah 5 bisa memakan waktu lebih lama & unduhan lebih besar dari sebelumnya.
- Kalau `python server.py` masih terasa macet/nggak merespons setelah klik **Mulai Kamera**: kemungkinan ada proses `server.py` lama dari sesi sebelumnya yang belum benar-benar mati dan masih memegang port 5050. Matikan dulu prosesnya (lihat cara di atas: `lsof`/`kill` untuk macOS, Task Manager untuk Windows) sebelum menjalankan ulang.
- **Video kamera hitam/gelap total setelah klik "Mulai Kamera"**: umum terjadi di laptop Windows yang punya kamera inframerah bawaan (dipakai Windows Hello) — kamera itu terdeteksi lebih dulu dari kamera utama dan tampak hitam total kalau tidak sedang dipakai login wajah. Server sudah otomatis mencoba beberapa kamera dan memilih yang gambarnya terang; kalau masih salah pilih, paksa index kamera manual sebelum menjalankan server:
  ```bat
  set SEMAPHORE_CAM_INDEX=1
  python server.py
  ```
  (macOS/Linux: `SEMAPHORE_CAM_INDEX=1 python server.py`) — coba angka 1, lalu 2, dst. sampai ketemu kamera yang benar.
- **LED kamera menyala tapi video tetap menampilkan "Kamera belum aktif"**: cek jendela CMD/terminal. Kalau muncul traceback `AttributeError: module 'mediapipe' has no attribute 'solutions'`, itu bug packaging MediaPipe yang dilaporkan banyak terjadi di Windows pada versi `mediapipe` yang lebih baru dari `0.10.14` (lihat [issue #6200](https://github.com/google-ai-edge/mediapipe/issues/6200) di repo resminya). `requirements.txt` sudah dipin ke `mediapipe==0.10.14` yang tidak kena bug ini — kalau masih muncul, pastikan versi lama sudah tidak nyangkut:
  ```bat
  pip uninstall mediapipe -y
  pip install -r requirements.txt
  ```
  Sejak perbaikan ini, kalaupun MediaPipe tetap gagal, aplikasi tidak lagi macet total — video & deteksi huruf semaphore tetap jalan, cuma gestur spasi/reset yang otomatis nonaktif (ada pesan `[!] MediaPipe Hands gagal...` di terminal).
