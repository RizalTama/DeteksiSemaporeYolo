# Cara Menjalankan di Windows — Runtut dari Awal

Panduan langkah demi langkah dari **ekstrak zip sampai aplikasi jalan** di Windows 10/11.

Ada 2 cara menjalankan perintah: **CMD** (bawaan Windows, disarankan) atau **Git Bash**.
Kerjakan dulu bagian **A (Persiapan)**, lalu pilih **SATU** saja: bagian **B (CMD)** atau **C (Git Bash)**. Ikuti nomornya berurutan, jangan ada yang dilompati.

---

## A. Persiapan Awal (wajib, cukup sekali)

**1. Ekstrak file zip**

1. Cari file `semaphore_client.zip` (misalnya di folder Downloads).
2. Klik kanan file tersebut → **Extract All...** → **Extract**.
3. Hasilnya adalah folder bernama **`semaphore`**.
4. Pindahkan folder `semaphore` itu ke `C:\` supaya alamatnya menjadi **`C:\semaphore`** (caranya: potong/cut folder `semaphore`, buka This PC → Local Disk (C:), lalu paste).

**2. Pastikan isi folder lengkap**

Buka `C:\semaphore`, harus ada file/folder berikut:

```
C:\semaphore\
├── app.py
├── requirements.txt
├── readme.md
├── run-windows.md
├── panduan.png
├── best2.pt
├── best2_int8_openvino_model\          <- model cepat (INT8)
├── yolov8n-pose.pt
└── yolov8n-pose_int8_openvino_model\   <- model pose cepat (INT8)
```

> ⚠️ Kalau ada yang kurang, ekstrak ulang zip-nya — jangan lanjut dulu.

**3. Install Python**

1. Download **Python 3.10, 3.11, atau 3.12** dari https://www.python.org/downloads/windows/
2. Jalankan installernya. **PENTING: centang "Add python.exe to PATH"** di layar pertama, baru klik **Install Now**.
3. Setelah selesai, tutup installernya.

**4. Cek Python berhasil terpasang**

1. Tekan `Win + R`, ketik `cmd`, Enter.
2. Ketik:

```bat
python --version
```

3. Harus muncul misalnya `Python 3.12.x`.
   Kalau muncul *"'python' is not recognized"* → ulangi langkah 3 dan pastikan centang **Add python.exe to PATH**.

Persiapan selesai. Sekarang pilih **B (CMD)** atau **C (Git Bash)**.

---

## B. Versi CMD (Command Prompt) — disarankan

**1. Buka CMD di dalam folder proyek**

Buka folder `C:\semaphore` di File Explorer → klik kolom alamat di bagian atas → ketik `cmd` → Enter.
(CMD akan terbuka dan tertulis `C:\semaphore>` — itu tandanya benar.)

**2. Buat virtual environment** (sekali saja)

```bat
python -m venv venv
```

**3. Aktifkan virtual environment**

```bat
venv\Scripts\activate
```

Berhasil kalau di awal baris muncul tulisan `(venv)`.

**4. Install dependensi** (sekali saja, download beberapa ratus MB — tunggu sampai selesai)

```bat
pip install -r requirements.txt
```

**5. Jalankan aplikasi**

```bat
python app.py
```

- Kalau Windows minta **izin kamera** → pilih **Allow/Izinkan**.
- Jendela **"YOLO Realtime"** terbuka menampilkan webcam.
- Baris pertama di CMD harus berbunyi:
  `Model deteksi: best2_int8_openvino_model | model pose: yolov8n-pose_int8_openvino_model`

**6. Menjalankan lagi di lain waktu** (langkah 2 dan 4 tidak perlu diulang)

```bat
cd C:\semaphore
venv\Scripts\activate
python app.py
```

Lanjut ke bagian **D. Cara Pakai**.

---

## C. Versi Git Bash

> Pakai cara ini hanya kalau Git Bash sudah terpasang (bawaan aplikasi Git for Windows).

**1. Buka Git Bash di dalam folder proyek**

Buka folder `C:\semaphore` di File Explorer → klik kanan di area kosong → **Git Bash Here**.
(Kalau menu itu tidak muncul, buka Git Bash dari Start Menu lalu ketik: `cd /c/semaphore`)

**2. Buat virtual environment** (sekali saja)

```bash
python -m venv venv
```

**3. Aktifkan virtual environment**

```bash
source venv/Scripts/activate
```

Berhasil kalau di awal baris muncul tulisan `(venv)`.

> ⚠️ Perhatikan: di Git Bash pakai `source venv/Scripts/activate` (garis miring biasa `/`),
> BUKAN `venv\Scripts\activate` seperti di CMD.

**4. Install dependensi** (sekali saja, download beberapa ratus MB — tunggu sampai selesai)

```bash
pip install -r requirements.txt
```

**5. Jalankan aplikasi**

```bash
python app.py
```

- Kalau Windows minta **izin kamera** → pilih **Allow/Izinkan**.
- Jendela **"YOLO Realtime"** terbuka menampilkan webcam.
- Baris pertama di terminal harus berbunyi:
  `Model deteksi: best2_int8_openvino_model | model pose: yolov8n-pose_int8_openvino_model`

**6. Menjalankan lagi di lain waktu** (langkah 2 dan 4 tidak perlu diulang)

```bash
cd /c/semaphore
source venv/Scripts/activate
python app.py
```

---

## D. Cara Pakai

1. Berdiri menghadap kamera, peragakan huruf semaphore sesuai **panduan.png**.
2. **Tahan pose sampai bar kuning penuh** → huruf masuk ke kalimat.
3. Huruf ganda (misal "LL"): turunkan bendera sebentar, lalu peragakan lagi.
4. Kalimat otomatis tersimpan di **hasil.txt**.

| Tombol | Fungsi |
|--------|--------|
| **S**  | Simpan gambar ke folder `output\` (setelah 4 gambar otomatis dibuat grid 2×2) |
| **R**  | Reset kalimat |
| **Q**  | Keluar |

---

## E. Kalau Ada Masalah

**"Webcam tidak bisa dibuka"**
→ Buka **Settings → Privacy & security → Camera**, pastikan "Camera access" dan "Let desktop apps access your camera" **On**. Tutup aplikasi lain yang sedang memakai kamera (Zoom, Teams, dll), lalu jalankan ulang.

**`python` tidak dikenali (CMD)**
→ Python belum masuk PATH. Install ulang Python dan centang **"Add python.exe to PATH"** (bagian A langkah 3).

**`bash: python: command not found` (Git Bash)**
→ Sama seperti di atas: install ulang Python dengan centang Add to PATH, lalu **tutup dan buka lagi** Git Bash. Kalau masih gagal, jalankan dengan `winpty python app.py`.

**`venv\Scripts\activate` tidak jalan di Git Bash**
→ Itu perintah CMD. Di Git Bash harus pakai: `source venv/Scripts/activate`.

**`pip install` gagal / lambat**
→ Pastikan koneksi internet stabil, lalu ulangi perintahnya. Kalau gagal di `torch`, jalankan dulu:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
kemudian ulangi `pip install -r requirements.txt`.

**Muncul `Model deteksi: best2.pt` (bukan `best2_int8_openvino_model`)**
→ Folder model INT8 tidak ikut terekstrak. Aplikasi tetap jalan tapi lebih lambat. Ekstrak ulang zip dan pastikan kedua folder `*_int8_openvino_model` ada (bagian A langkah 2).

**Jendela kamera terbuka tapi gambarnya hitam/gelap total**
→ Kemungkinan besar aplikasi salah pilih kamera: banyak laptop Windows punya kamera inframerah bawaan (dipakai untuk Windows Hello) yang terdeteksi *sebelum* kamera utama, dan kamera IR itu tampak hitam total kalau tidak sedang dipakai untuk login wajah. Aplikasi sudah otomatis mencoba beberapa kamera dan memilih yang gambarnya terang, tapi kalau masih salah pilih juga, paksa index kamera tertentu secara manual:
```bat
set SEMAPHORE_CAM_INDEX=1
python app.py
```
(coba angka 1, lalu 2, dst. sampai ketemu kamera yang benar — jalankan `python app.py` ulang tiap ganti angka).

**Deteksi terasa lambat menangkap huruf**
→ Buka `app.py`, ubah `MIN_VOTES = 9` menjadi `6` dan `WINDOW = 12` menjadi `8`.

**Banyak huruf salah masuk**
→ Kebalikannya: naikkan `MIN_VOTES` (misal 10) dan pastikan pencahayaan cukup serta seluruh badan + bendera masuk frame kamera.
