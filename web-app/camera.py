"""Background capture loop: buka webcam sekali, jalankan deteksi tiap frame,
simpan hasil ke state bersama (dilindungi lock) supaya bisa dibaca oleh
banyak request Flask (/video_feed, /status) tanpa membuka kamera berkali-kali.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from detector import MIN_VOTES, SemaphoreDetector, StabilisatorHuruf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
HASIL_PATH = os.path.join(BASE_DIR, "hasil.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# gestur "berdiri tegap" (netral) -> tambah spasi
SPASI_DURASI = 7.5    # detik posisi netral tertahan sebelum spasi ditambahkan
SPASI_COOLDOWN = 2.5  # jeda setelah spasi ditambahkan sebelum timer boleh mulai lagi
RESET_FLASH_DURASI = 1.5  # lama teks "Reset!" ditampilkan di UI setelah gestur reset

MAX_KAMERA_INDEX = 4      # jumlah indeks kamera yang dicoba sebelum menyerah
FRAME_GELAP_AMBANG = 15.0  # rata-rata pixel (0-255) di bawah ini dianggap frame hitam


def _buka_kamera():
    """Coba beberapa indeks kamera dan pilih yang menghasilkan frame terang.

    Banyak laptop Windows mengekspos kamera inframerah (dipakai Windows Hello)
    di indeks sebelum kamera RGB utama -- index 0 bisa saja terbuka sukses
    (isOpened() == True) tapi framenya hitam total karena tidak ada iluminator
    IR aktif. Kalau semua indeks yang dicoba gelap, kembalikan kandidat
    pertama yang berhasil dibuka (perilaku lama) supaya tetap ada fallback,
    bukan error total.

    Set env var SEMAPHORE_CAM_INDEX untuk memaksa indeks tertentu kalau
    auto-deteksi ini salah tebak.
    """
    index_manual = os.environ.get("SEMAPHORE_CAM_INDEX")
    if index_manual is not None:
        idx = int(index_manual)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
        return cap if cap.isOpened() else None

    kandidat_gelap = None
    for idx in range(MAX_KAMERA_INDEX):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > FRAME_GELAP_AMBANG:
                return cap

        if kandidat_gelap is None:
            kandidat_gelap = cap
        else:
            cap.release()

    return kandidat_gelap


class CameraStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.cap = None
        self.thread = None
        self.running = False
        self.detector = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # untuk YOLO det+pose; MediaPipe Hands punya executor sendiri di SemaphoreDetector

        self.stab = StabilisatorHuruf()
        self.saved_images = []
        self._grid_ready = False

        self.latest_jpeg = None
        self._last_annotated = None

        self.netral_start = None
        self.spasi_cooldown_until = 0.0
        self.reset_flash_until = 0.0

        self.status = {
            "running": False,
            "huruf_yolo": None,
            "conf": 0.0,
            "huruf_pose": None,
            "kandidat": None,
            "suara": 0,
            "min_votes": MIN_VOTES,
            "kalimat": "",
            "error": None,
            "spasi_progress": 0.0,
            "spasi_cooldown": False,
            "reset_flash": False,
        }

    def start(self):
        with self.lock:
            if self.running:
                return True, None
            if self.detector is None:
                self.detector = SemaphoreDetector()
            self.cap = _buka_kamera()
            if self.cap is None or not self.cap.isOpened():
                self.cap = None
                self.status["error"] = "Webcam tidak bisa dibuka"
                return False, "Webcam tidak bisa dibuka"
            self.running = True
            self.status["error"] = None
            self.status["running"] = True

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True, None

    def stop(self):
        with self.lock:
            self.running = False
            self.status["running"] = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.latest_jpeg = None
            self._last_annotated = None
            self.netral_start = None
            self.spasi_cooldown_until = 0.0
            self.reset_flash_until = 0.0
            self.status.update({
                "huruf_yolo": None,
                "conf": 0.0,
                "huruf_pose": None,
                "kandidat": None,
                "suara": 0,
                "spasi_progress": 0.0,
                "spasi_cooldown": False,
                "reset_flash": False,
            })

    def _loop(self):
        while True:
            with self.lock:
                if not self.running or self.cap is None:
                    break
                cap = self.cap

            ret, frame = cap.read()
            if not ret:
                with self.lock:
                    self.status["error"] = "Gagal membaca frame webcam"
                    self.running = False
                    self.status["running"] = False
                break

            (annotated, huruf, huruf_yolo, conf, huruf_pose, _,
             netral, reset_gesture) = self.detector.process(frame, self.executor)

            now = time.monotonic()
            kalimat_changed = False

            if reset_gesture:
                # gestur dua jari: reset langsung, prioritas di atas gestur spasi
                self.stab.reset()
                self.netral_start = None
                self.spasi_cooldown_until = now + SPASI_COOLDOWN
                self.reset_flash_until = now + RESET_FLASH_DURASI
                kalimat_changed = True

            baru, kandidat, suara = self.stab.update(huruf)
            if baru:
                kalimat_changed = True

            spasi_progress = 0.0
            if not reset_gesture:
                in_cooldown = now < self.spasi_cooldown_until
                if netral and not in_cooldown:
                    if self.netral_start is None:
                        self.netral_start = now
                    elapsed = now - self.netral_start
                    if elapsed >= SPASI_DURASI:
                        self.stab.tambah_spasi()
                        self.netral_start = None
                        self.spasi_cooldown_until = now + SPASI_COOLDOWN
                        kalimat_changed = True
                    else:
                        spasi_progress = elapsed / SPASI_DURASI
                else:
                    self.netral_start = None

            if kalimat_changed:
                with open(HASIL_PATH, "w") as f:
                    f.write(self.stab.kalimat)

            ok, buf = cv2.imencode(".jpg", annotated)
            with self.lock:
                if ok:
                    self.latest_jpeg = buf.tobytes()
                    self._last_annotated = annotated
                self.status.update({
                    "huruf_yolo": huruf_yolo,
                    "conf": round(conf, 2),
                    "huruf_pose": huruf_pose,
                    "kandidat": kandidat,
                    "suara": suara,
                    "kalimat": self.stab.kalimat,
                    "spasi_progress": round(spasi_progress, 2),
                    "spasi_cooldown": now < self.spasi_cooldown_until,
                    "reset_flash": now < self.reset_flash_until,
                })

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_status(self):
        with self.lock:
            status = dict(self.status)
            status["saved_count"] = min(len(self.saved_images), 4)
            status["grid_ready"] = self._grid_ready
            return status

    def save_frame(self):
        with self.lock:
            if self._last_annotated is None:
                return False, "Belum ada frame untuk disimpan", None
            frame_to_save = self._last_annotated.copy()
            kalimat = self.stab.kalimat

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deteksi_{timestamp}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, frame_to_save)

        with self.lock:
            self.saved_images.append(save_path)
            count = len(self.saved_images)

        grid_created = False
        if count >= 4:
            self._buat_grid(kalimat)
            with self.lock:
                self._grid_ready = True
            grid_created = True

        return True, filename, {"count": min(count, 4), "grid_created": grid_created}

    def _buat_grid(self, kalimat):
        with self.lock:
            paths = self.saved_images[-4:]
        imgs = [cv2.imread(p) for p in paths]

        h, w = imgs[0].shape[:2]
        imgs = [cv2.resize(im, (w, h)) for im in imgs]

        row1 = np.hstack((imgs[0], imgs[1]))
        row2 = np.hstack((imgs[2], imgs[3]))
        grid = np.vstack((row1, row2))

        cv2.putText(grid, f"KALIMAT: {kalimat}", (20, grid.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        grid_path = os.path.join(OUTPUT_DIR, "grid_final.jpg")
        cv2.imwrite(grid_path, grid)

    def reset(self):
        with self.lock:
            self.stab.reset()
            self.saved_images = []
            self._grid_ready = False
            self.netral_start = None
            self.spasi_cooldown_until = 0.0
            self.reset_flash_until = 0.0
            self.status.update({
                "kalimat": "",
                "kandidat": None,
                "suara": 0,
                "spasi_progress": 0.0,
                "spasi_cooldown": False,
                "reset_flash": False,
            })
