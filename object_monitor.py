"""
Object Monitor (Alat Monitoring Barang)
------------------------------------------
Alat ini BEDA dari object_detector.py biasa: alat ini terus memantau
kamera dari waktu ke waktu, MENCATAT LOG setiap barang yang terdeteksi
ke file CSV, memberi notifikasi visual saat ada barang baru muncul atau
menghilang dari pandangan kamera, dan membuat RINGKASAN sesi saat program
ditutup.

Menggunakan MediaPipe Tasks API (ObjectDetector) dengan model EfficientDet-Lite0.

Cara pakai:
    python object_monitor.py

Tekan 'q' untuk keluar (ringkasan sesi akan dicetak ke terminal).
Tekan 'c' untuk ganti kamera.
Tekan 's' untuk simpan screenshot frame saat ini ke folder snapshots/.

Catatan: saat pertama kali dijalankan, program otomatis mendownload file
model AI (efficientdet_lite0.tflite, sekitar 4-5 MB). Pastikan ada koneksi
internet saat pertama kali run.

Output:
- monitoring_log.csv  -> log lengkap tiap barang terdeteksi per waktu
- snapshots/          -> folder screenshot (kalau tombol 's' ditekan)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import time
import csv
from datetime import datetime
from collections import defaultdict


# ---------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------
CAMERA_INDEX = 1

MODEL_PATH = "efficientdet_lite0.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)

SCORE_THRESHOLD = 0.5
MAX_RESULTS = 8

LOG_FILE = "monitoring_log.csv"
SNAPSHOT_DIR = "snapshots"

# Setiap berapa detik sekali status barang dicatat ke CSV (biar file tidak
# kebanjiran ribuan baris per detik)
LOG_INTERVAL_SECONDS = 2.0

# Berapa lama (detik) sebuah barang harus HILANG dari layar sebelum dianggap
# benar-benar "hilang" (menghindari notifikasi palsu karena deteksi kedip sesaat)
DISAPPEAR_GRACE_SECONDS = 1.5


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print("[INFO] Model belum ada, mendownload efficientdet_lite0.tflite ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OK] Model berhasil didownload.")
    except Exception as e:
        print(f"[ERROR] Gagal download model: {e}")
        print("Cek koneksi internet kamu, lalu coba jalankan lagi.")
        raise SystemExit(1)


def create_object_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        score_threshold=SCORE_THRESHOLD,
        max_results=MAX_RESULTS,
    )
    return mp_vision.ObjectDetector.create_from_options(options)


def open_camera(preferred_index):
    cap = cv2.VideoCapture(preferred_index)
    if cap.isOpened():
        return cap, preferred_index

    print(f"[!] Kamera index {preferred_index} tidak ditemukan, mencoba index lain...")
    for i in range(0, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[OK] Menggunakan kamera index {i}")
            return cap, i
    return None, -1


class ObjectMonitor:
    """Mengelola state monitoring: barang apa yang sedang terlihat,
    kapan terakhir terlihat, total kemunculan, dan histori notifikasi."""

    def __init__(self):
        self.currently_visible = {}   # label -> waktu terakhir terlihat
        self.total_seen_count = defaultdict(int)   # label -> berapa kali "muncul" (event baru)
        self.first_seen_time = {}     # label -> waktu pertama kali terlihat sepanjang sesi
        self.notifications = []       # list (waktu, pesan) untuk ditampilkan sebentar di layar
        self.last_log_time = 0
        self.session_start = time.time()

        self._init_log_file()

    def _init_log_file(self):
        file_exists = os.path.exists(LOG_FILE)
        self.log_fp = open(LOG_FILE, "a", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_fp)
        if not file_exists:
            self.log_writer.writerow(["timestamp", "label", "score", "x", "y", "width", "height"])

    def update(self, detections):
        now = time.time()
        seen_this_frame = set()

        for detection in detections:
            label = detection.categories[0].category_name
            score = detection.categories[0].score
            bbox = detection.bounding_box

            seen_this_frame.add(label)

            # Barang baru muncul (belum ada di currently_visible, atau sudah lama hilang)
            if label not in self.currently_visible:
                self.total_seen_count[label] += 1
                self.first_seen_time.setdefault(label, now)
                self._add_notification(f"+ {label} terdeteksi")

            self.currently_visible[label] = now

            # Log ke CSV berdasarkan interval, biar tidak kebanjiran data
            if now - self.last_log_time >= LOG_INTERVAL_SECONDS:
                self.log_writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    label, f"{score:.2f}",
                    bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                ])

        if now - self.last_log_time >= LOG_INTERVAL_SECONDS:
            self.log_fp.flush()
            self.last_log_time = now

        # Cek barang yang sudah lama tidak terlihat -> anggap hilang
        hilang = []
        for label, last_time in list(self.currently_visible.items()):
            if label not in seen_this_frame and (now - last_time) > DISAPPEAR_GRACE_SECONDS:
                hilang.append(label)

        for label in hilang:
            del self.currently_visible[label]
            self._add_notification(f"- {label} hilang dari pandangan")

    def _add_notification(self, message):
        self.notifications.append((time.time(), message))
        print(f"[EVENT] {message}")

    def get_active_notifications(self, display_seconds=3.0):
        now = time.time()
        self.notifications = [(t, m) for (t, m) in self.notifications if now - t < display_seconds]
        return [m for (_, m) in self.notifications]

    def print_summary(self):
        duration = time.time() - self.session_start
        print("\n" + "=" * 50)
        print("RINGKASAN SESI MONITORING")
        print("=" * 50)
        print(f"Durasi monitoring : {duration/60:.1f} menit ({duration:.0f} detik)")
        print(f"Jenis barang unik terdeteksi : {len(self.total_seen_count)}")
        print("-" * 50)
        if self.total_seen_count:
            for label, count in sorted(self.total_seen_count.items(), key=lambda x: -x[1]):
                first_seen = self.first_seen_time[label] - self.session_start
                print(f"  - {label:20s} muncul {count}x  (pertama kali di {first_seen:.0f}s)")
        else:
            print("  Tidak ada barang yang terdeteksi selama sesi ini.")
        print("=" * 50)
        print(f"Log lengkap tersimpan di: {os.path.abspath(LOG_FILE)}")

    def close(self):
        self.log_fp.close()


def draw_overlay(frame, detections, monitor, cam_index, fps):
    h, w, _ = frame.shape

    # Kotak + label untuk tiap barang terdeteksi
    for detection in detections:
        bbox = detection.bounding_box
        x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        category = detection.categories[0]
        label, score = category.category_name, category.score

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), (0, 255, 0), -1)
        cv2.putText(frame, text, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Panel info kiri atas
    cv2.rectangle(frame, (10, 10), (330, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Sedang terpantau: {len(monitor.currently_visible)} barang", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}  |  Kamera: {cam_index}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Notifikasi event terbaru (muncul/hilang), tampil sementara di kanan atas
    active_notifs = monitor.get_active_notifications()
    ny = 30
    for msg in active_notifs[-5:]:
        color = (0, 255, 255) if msg.startswith("+") else (0, 100, 255)
        cv2.putText(frame, msg, (w - 350, ny),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        ny += 30


def main():
    ensure_model_downloaded()
    detector = create_object_detector()
    monitor = ObjectMonitor()

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    cap, cam_index = open_camera(CAMERA_INDEX)
    if cap is None:
        print("[ERROR] Tidak ada kamera yang terdeteksi. Cek koneksi USB kamera eksternal.")
        monitor.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0
    start_time = time.time()

    print("Monitoring dimulai. Tekan 'q' untuk berhenti, 'c' ganti kamera, 's' simpan snapshot.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[ERROR] Gagal membaca frame dari kamera.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)
            detections = result.detections if result.detections else []

            monitor.update(detections)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            draw_overlay(frame, detections, monitor, cam_index, fps)
            cv2.imshow("Object Monitor - Alat Monitoring Barang", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                cap.release()
                cam_index = (cam_index + 1) % 5
                cap, cam_index = open_camera(cam_index)
                if cap is None:
                    print("[ERROR] Tidak ada kamera lain yang ditemukan.")
                    break
            elif key == ord('s'):
                filename = os.path.join(
                    SNAPSHOT_DIR, f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(filename, frame)
                print(f"[INFO] Snapshot disimpan: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        monitor.print_summary()
        monitor.close()


if __name__ == "__main__":
    main()