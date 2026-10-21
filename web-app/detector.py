"""Logika deteksi semaphore, di-port dari app.py (desktop) tanpa mengubah logika.

Model diakses dari folder root project (bukan di-copy ke web-app/) supaya
tidak ada duplikasi file model yang besar.
"""

import math
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pilih_model(folder_int8, file_pt):
    path_int8 = os.path.join(ROOT_DIR, folder_int8)
    if os.path.isdir(path_int8):
        return path_int8
    return os.path.join(ROOT_DIR, file_pt)


DET_MODEL = _pilih_model("best2_int8_openvino_model", "best2.pt")
POSE_MODEL = _pilih_model("yolov8n-pose_int8_openvino_model", "yolov8n-pose.pt")

CONF = 0.15        # ambang deteksi (rendah tidak apa-apa, kestabilan dijaga voting)
IOU = 0.4
IMGSZ = 832        # harus sama dengan ukuran gambar saat training
POSE_IMGSZ = 640
WINDOW = 16        # panjang jendela voting (frame) -- dinaikkan dari 12 (versi desktop) biar pose harus ditahan sedikit lebih lama
MIN_VOTES = 12     # huruf diterima jika jadi mayoritas >= 12 dari 16 frame terakhir
GAP_RESET = 8      # frame kosong berturut-turut -> huruf yang sama boleh diulang

# gestur tambahan (spasi & reset), lihat CameraStream untuk timer/cooldown-nya
HANDS_CONF = 0.5     # ambang deteksi & tracking MediaPipe Hands
# tiap jari dicek dari sudut di sendi PIP: (MCP, PIP, TIP). Gestur reset = "2 jari"
# (telunjuk+tengah lurus, manis+kelingking menekuk, mirip simbol victori/peace) --
# dipilih ketimbang "5 jari terbuka" karena jauh lebih jarang kebentuk tanpa sengaja
# di sela gerakan transisi antar huruf semaphore (ibu jari tidak dicek/bebas).
_JARI_LURUS = [(5, 6, 8), (9, 10, 12)]      # telunjuk, tengah
_JARI_TEKUK = [(13, 14, 16), (17, 18, 20)]  # manis, kelingking
FINGER_STRAIGHT_ANGLE = 160  # sudut minimum (derajat) di PIP supaya jari dianggap lurus

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

    def tambah_spasi(self):
        """Tambah spasi ke kalimat (dipicu gestur netral tertahan), di luar
        jalur voting huruf biasa."""
        self.kalimat += " "


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


def deteksi_netral(result):
    """Deteksi posisi netral (kedua lengan turun, sektor 'D') dari hasil pose YOLO
    yang sama dipakai huruf_dari_pose -- dipakai sebagai pemicu gestur spasi.

    Pakai gerbang visibilitas yang sama (bahu, pergelangan, pinggul harus
    terlihat jelas) supaya konsisten dengan deteksi huruf, tapi fungsi ini
    terpisah dari huruf_dari_pose supaya logika huruf yang sudah teruji tidak
    ikut berubah.
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return False
    kp = result.keypoints.xy[0].tolist()
    if len(kp) < 13 or result.keypoints.conf is None:
        return False
    cf = result.keypoints.conf[0].tolist()
    if min(cf[5], cf[6], cf[9], cf[10]) < 0.5 or min(cf[11], cf[12]) < 0.5:
        return False
    for s_i, w_i in ((6, 10), (5, 9)):
        sx, sy = kp[s_i]
        wx, wy = kp[w_i]
        if min(sx, sy, wx, wy) <= 0:
            return False
        if sektor_lengan(sx, sy, wx, wy) != "D":
            return False
    return True


def _sudut(a, b, c):
    """Sudut di titik b yang dibentuk oleh garis b-a dan b-c, dalam derajat."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 <= 1e-6 or mag2 <= 1e-6:
        return 0.0
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


def dua_jari_terbuka(hand_landmarks):
    """Deteksi gestur reset 'dua jari' (telunjuk+tengah lurus, manis+kelingking
    menekuk, mirip simbol victori/peace) dari 21 landmark MediaPipe Hands.

    Dipilih ketimbang 'lima jari terbuka': gestur itu ternyata gampang
    kebentuk tanpa sengaja di sela gerakan transisi antar huruf semaphore,
    dan juga sempat salah kebaca saat menggenggam tongkat bendera (jari yang
    melingkar di tongkat menekuk jelas di sendi tengah, tapi diameter
    tongkat menahan ujungnya supaya tidak sampai serapat kepalan penuh).
    'Dua jari' jauh lebih jarang terbentuk tanpa sengaja karena butuh dua
    jari spesifik lurus SEKALIGUS dua jari lain menekuk -- kombinasi yang
    hampir tidak mungkin muncul selagi tangan terus menggenggam bendera.
    Kelurusan tiap jari dicek dari sudut di sendi PIP-nya (MCP-PIP-TIP),
    supaya tidak terpengaruh arah hadap tangan di kamera. Ibu jari tidak
    dicek/dibiarkan bebas.
    """
    lm = hand_landmarks.landmark

    for mcp_i, pip_i, tip_i in _JARI_LURUS:
        if _sudut(lm[mcp_i], lm[pip_i], lm[tip_i]) < FINGER_STRAIGHT_ANGLE:
            return False

    for mcp_i, pip_i, tip_i in _JARI_TEKUK:
        if _sudut(lm[mcp_i], lm[pip_i], lm[tip_i]) >= FINGER_STRAIGHT_ANGLE:
            return False

    return True


def gabung(huruf_pose, huruf_yolo):
    """Fusi: dekoder pose menang saat menghasilkan huruf valid, YOLO mengisi sisanya.
    (Tervalidasi di foto klien: yolo 20/26, pose 23/26, fusi 24/26.)"""
    return huruf_pose if huruf_pose else huruf_yolo


class SemaphoreDetector:
    """Membungkus kedua model YOLO + MediaPipe Hands + fusi logic supaya bisa
    dipanggil satu method per frame."""

    def __init__(self):
        self.det = YOLO(DET_MODEL, task="detect")
        self.pose = YOLO(POSE_MODEL, task="pose")

        # Warm-up sekuensial (bukan lewat executor): binding Python OpenVINO
        # (pybind11) punya inisialisasi numpy dtype API yang cuma jalan sekali
        # (std::call_once). Kalau dipicu dari 2 thread bersamaan -- persis yang
        # terjadi kalau det & pose langsung dipanggil paralel lewat
        # ThreadPoolExecutor saat frame pertama -- pernah macet total (deadlock
        # rebutan GIL di dalam call_once). Panggil predict() sekali di sini dulu
        # untuk masing-masing model supaya inisialisasi itu selesai aman di 1
        # thread sebelum executor dipakai.
        _dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        self.det.predict(source=_dummy, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)
        self.pose.predict(source=_dummy, conf=0.3, imgsz=POSE_IMGSZ, verbose=False)

        # MediaPipe Hands (Solutions API) TIDAK thread-safe -- objeknya harus dibuat
        # dan dipanggil .process() dari satu thread yang sama terus-menerus, kalau
        # tidak bisa hang (pernah terjadi di macOS, kemungkinan besar karena GL
        # context di dalam MediaPipe yang thread-affine). Karena itu Hands dibuat
        # & dipanggil lewat 1 executor khusus (max_workers=1) yang selalu memakai
        # worker thread yang sama, terpisah dari executor YOLO det+pose.
        self._hands_executor = ThreadPoolExecutor(max_workers=1)
        self.hands = None
        self._hands_gagal = False  # True kalau MediaPipe Hands error & fitur gestur dimatikan

    def _init_dan_deteksi_tangan(self, frame):
        """Gestur tangan (spasi/reset) itu fitur tambahan, bukan fitur inti.
        Kalau MediaPipe gagal (mis. instalasi Windows yang kena bug packaging
        'module mediapipe has no attribute solutions' -- lihat
        github.com/google-ai-edge/mediapipe/issues/6200), fitur ini dimatikan
        sendiri dan sisanya (deteksi huruf semaphore) tetap jalan, alih-alih
        mematikan seluruh thread capture."""
        if self._hands_gagal:
            return False
        try:
            if self.hands is None:
                self.hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=HANDS_CONF,
                    min_tracking_confidence=HANDS_CONF,
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hasil = self.hands.process(rgb)
        except Exception as e:
            print(f"[!] MediaPipe Hands gagal, gestur spasi/reset dimatikan: {e}", file=sys.stderr)
            self._hands_gagal = True
            return False
        if not hasil.multi_hand_landmarks:
            return False
        return any(dua_jari_terbuka(lm) for lm in hasil.multi_hand_landmarks)

    def process(self, frame, executor):
        f_det = executor.submit(self.det.predict, source=frame, conf=CONF, iou=IOU,
                                 imgsz=IMGSZ, verbose=False)
        f_pose = executor.submit(self.pose.predict, source=frame, conf=0.3,
                                  imgsz=POSE_IMGSZ, verbose=False)
        f_hands = self._hands_executor.submit(self._init_dan_deteksi_tangan, frame)
        r_det = f_det.result()[0]
        r_pose = f_pose.result()[0]
        reset_gesture = f_hands.result()

        annotated_frame = r_det.plot()

        huruf_yolo, conf = deteksi_terbaik(r_det, self.det.names)
        huruf_pose = huruf_dari_pose(r_pose)
        huruf = gabung(huruf_pose, huruf_yolo)
        netral = deteksi_netral(r_pose)
        num_objects = len(r_det.boxes) if r_det.boxes is not None else 0

        return (annotated_frame, huruf, huruf_yolo, conf, huruf_pose, num_objects,
                netral, reset_gesture)
