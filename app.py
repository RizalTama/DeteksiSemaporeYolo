from ultralytics import YOLO
import cv2
import os
import sys
import math
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# model OpenVINO INT8 (cepat); fallback ke .pt kalau folder ekspor tidak ada
DET_MODEL = "best2_int8_openvino_model" if os.path.isdir("best2_int8_openvino_model") else "best2.pt"
POSE_MODEL = ("yolov8n-pose_int8_openvino_model"
              if os.path.isdir("yolov8n-pose_int8_openvino_model") else "yolov8n-pose.pt")

CONF = 0.15        # ambang deteksi (rendah tidak apa-apa, kestabilan dijaga voting)
IOU = 0.4
IMGSZ = 832        # harus sama dengan ukuran gambar saat training
POSE_IMGSZ = 640
WINDOW = 12        # panjang jendela voting (frame)
MIN_VOTES = 9      # huruf diterima jika jadi mayoritas >= 9 dari 12 frame terakhir
GAP_RESET = 8      # frame kosong berturut-turut -> huruf yang sama boleh diulang

MAX_KAMERA_INDEX = 4      # jumlah indeks kamera yang dicoba sebelum menyerah
FRAME_GELAP_AMBANG = 15.0  # rata-rata pixel (0-255) di bawah ini dianggap frame hitam

# tabel semaphore (perspektif penonton) sesuai panduan.png tervalidasi.
# posisi lengan: U=atas, D=bawah, Lh/Lo/Ll = kiri tinggi/datar/rendah, Rh/Ro/Rl = kanan
TABEL_SEMAPHORE = {
    frozenset(["Ll", "D"]): "A", frozenset(["Lo", "D"]): "B", frozenset(["Lh", "D"]): "C",
    frozenset(["U", "D"]): "D",  frozenset(["Rh", "D"]): "E", frozenset(["Ro", "D"]): "F",
    frozenset(["Rl", "D"]): "G", frozenset(["Lo", "Ll"]): "H", frozenset(["Lh", "Ll"]): "I",
    frozenset(["U", "Ro"]): "J", frozenset(["U", "Ll"]): "K", frozenset(["Lh", "Rl"]): "L",
    frozenset(["Ro", "Ll"]): "M", frozenset(["Ll", "Rl"]): "N", frozenset(["Lh", "Lo"]): "O",
    frozenset(["U", "Lo"]): "P", frozenset(["Rh", "Lo"]): "Q", frozenset(["Lo", "Ro"]): "R",
    frozenset(["Lo", "Rl"]): "S", frozenset(["U", "Lh"]): "T", frozenset(["Lh", "Rh"]): "U",
    frozenset(["U", "Rl"]): "V", frozenset(["Rh", "Ro"]): "W", frozenset(["Rh", "Rl"]): "X",
    frozenset(["Lh", "Ro"]): "Y", frozenset(["Ro", "Rl"]): "Z",
}


class StabilisatorHuruf:
    """Menyaring deteksi per-frame menjadi huruf yang stabil.

    Huruf hanya ditambahkan ke kalimat jika menjadi mayoritas dalam
    jendela voting. Huruf yang sama bisa diulang setelah bendera
    diturunkan (beberapa frame tanpa deteksi).
    """

    def __init__(self, window=WINDOW, min_votes=MIN_VOTES, gap_reset=GAP_RESET):
        self.votes = deque(maxlen=window)
        self.min_votes = min_votes
        self.gap_reset = gap_reset
        self.last_accepted = None
        self.empty_streak = 0
        self.kalimat = ""

    def update(self, huruf):
        """Terima huruf terdeteksi frame ini (atau None). Mengembalikan
        (huruf_baru_atau_None, kandidat, jumlah_suara)."""
        self.votes.append(huruf)

        if huruf is None:
            self.empty_streak += 1
            if self.empty_streak >= self.gap_reset:
                self.last_accepted = None
        else:
            self.empty_streak = 0

        letters = [h for h in self.votes if h]
        if not letters:
            return None, None, 0

        kandidat = max(set(letters), key=letters.count)
        suara = letters.count(kandidat)

        if suara >= self.min_votes and kandidat != self.last_accepted:
            self.kalimat += kandidat
            self.last_accepted = kandidat
            self.votes.clear()
            return kandidat, kandidat, suara

        return None, kandidat, suara

    def reset(self):
        self.kalimat = ""
        self.last_accepted = None
        self.empty_streak = 0
        self.votes.clear()


def deteksi_terbaik(result, names):
    """Ambil huruf dari kotak dengan confidence tertinggi."""
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0
    i = int(result.boxes.conf.argmax())
    return names[int(result.boxes.cls[i])], float(result.boxes.conf[i])


def sektor_lengan(sx, sy, wx, wy):
    """Klasifikasi arah bahu->pergelangan ke 8 sektor kompas (koordinat gambar)."""
    ang = math.degrees(math.atan2(sy - wy, wx - sx))  # 0=kanan penonton, 90=atas
    s = round(ang / 45) * 45
    if s == -180:
        s = 180
    return {90: "U", -90: "D", 45: "Rh", 0: "Ro", -45: "Rl",
            135: "Lh", 180: "Lo", -135: "Ll"}[s]


def huruf_dari_pose(result):
    """Dekode huruf semaphore dari sudut kedua lengan (keypoints COCO).

    Hanya aktif jika orangnya berdiri utuh di frame: bahu, pergelangan,
    dan PINGGUL harus terlihat jelas. Tanpa gerbang ini, orang yang duduk
    di depan laptop (tangan di meja/keyboard) ikut terbaca sebagai huruf.
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return None
    kp = result.keypoints.xy[0].tolist()
    if len(kp) < 13 or result.keypoints.conf is None:
        return None
    cf = result.keypoints.conf[0].tolist()
    if min(cf[5], cf[6], cf[9], cf[10]) < 0.5 or min(cf[11], cf[12]) < 0.5:
        return None
    pos = set()
    for s_i, w_i in ((6, 10), (5, 9)):   # bahu->pergelangan kanan & kiri (COCO)
        sx, sy = kp[s_i]
        wx, wy = kp[w_i]
        if min(sx, sy, wx, wy) <= 0:     # keypoint tidak terlihat
            return None
        pos.add(sektor_lengan(sx, sy, wx, wy))
    return TABEL_SEMAPHORE.get(frozenset(pos))


def gabung(huruf_pose, huruf_yolo):
    """Fusi: dekoder pose menang saat menghasilkan huruf valid, YOLO mengisi sisanya.
    (Tervalidasi di foto klien: yolo 20/26, pose 23/26, fusi 24/26.)"""
    return huruf_pose if huruf_pose else huruf_yolo


def buka_kamera():
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


def main():
    det = YOLO(DET_MODEL, task="detect")
    pose = YOLO(POSE_MODEL, task="pose")
    print("Model deteksi:", DET_MODEL, "| model pose:", POSE_MODEL)

    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)

    cap = buka_kamera()
    if cap is None or not cap.isOpened():
        print("[X] Webcam tidak bisa dibuka!")
        return

    print("Tekan 'S' = Save gambar")
    print("Tekan 'R' = Reset kalimat")
    print("Tekan 'Q' = Keluar")
    print("Tips: tahan pose sampai bar penuh; turunkan bendera untuk huruf ganda")

    stab = StabilisatorHuruf()
    saved_images = []
    executor = ThreadPoolExecutor(max_workers=2)   # deteksi & pose jalan paralel

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        f_det = executor.submit(det.predict, source=frame, conf=CONF, iou=IOU,
                                imgsz=IMGSZ, verbose=False)
        f_pose = executor.submit(pose.predict, source=frame, conf=0.3,
                                 imgsz=POSE_IMGSZ, verbose=False)
        r_det = f_det.result()[0]
        r_pose = f_pose.result()[0]

        annotated_frame = r_det.plot()

        huruf_yolo, conf = deteksi_terbaik(r_det, det.names)
        huruf_pose = huruf_dari_pose(r_pose)
        huruf = gabung(huruf_pose, huruf_yolo)
        baru, kandidat, suara = stab.update(huruf)

        if baru:
            with open("hasil.txt", "w") as f:
                f.write(stab.kalimat)

        num_objects = len(r_det.boxes) if r_det.boxes is not None else 0
        info = f"YOLO: {huruf_yolo or '-'} {conf:.2f} | Pose: {huruf_pose or '-'}"
        cv2.putText(annotated_frame, info, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(annotated_frame, f"Kalimat: {stab.kalimat}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # indikator kestabilan: huruf kandidat + bar kemajuan voting
        if kandidat:
            cv2.putText(annotated_frame, f"Menahan: {kandidat}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            lebar = int(200 * min(suara / MIN_VOTES, 1.0))
            cv2.rectangle(annotated_frame, (240, 120), (240 + lebar, 145),
                          (0, 255, 255), -1)
            cv2.rectangle(annotated_frame, (240, 120), (440, 145),
                          (255, 255, 255), 2)

        cv2.imshow("YOLO Realtime", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deteksi_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, annotated_frame)

            saved_images.append(save_path)
            print("Gambar disimpan:", save_path)

            if len(saved_images) >= 4:   # minimal 4 gambar untuk grid
                imgs = [cv2.imread(p) for p in saved_images]

                # samakan ukuran
                h, w = imgs[0].shape[:2]
                imgs = [cv2.resize(im, (w, h)) for im in imgs]

                # buat grid 2x2
                row1 = np.hstack((imgs[0], imgs[1]))
                row2 = np.hstack((imgs[2], imgs[3]))
                grid = np.vstack((row1, row2))

                cv2.putText(grid, f"KALIMAT: {stab.kalimat}",
                            (20, grid.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255), 3)

                grid_path = os.path.join(save_dir, "grid_final.jpg")
                cv2.imwrite(grid_path, grid)
                print("Grid final disimpan:", grid_path)

        elif key == ord('r'):
            stab.reset()
            saved_images = []
            print("Kalimat & gambar reset")

        elif key == ord('q'):
            break

    executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
