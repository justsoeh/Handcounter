"""
Hand Finger Counter
--------------------
Mendeteksi tangan lewat kamera (termasuk kamera eksternal / webcam USB)
dan menghitung jumlah jari yang terangkat (1-5).

Menggunakan MediaPipe Tasks API (HandLandmarker) -- API baru yang
menggantikan mp.solutions.hands yang lama (yang sudah rusak/deprecated
di versi mediapipe terbaru, 0.10.30+).

Cara pakai:
    python main.py

Tekan 'q' untuk keluar.
Tekan 'c' untuk ganti kamera (jika kamera eksternal tidak terdeteksi otomatis).

Catatan: saat pertama kali dijalankan, program akan otomatis mendownload
file model AI (hand_landmarker.task, sekitar 8 MB) dari internet.
Pastikan ada koneksi internet saat pertama kali run.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import time


# ---------------------------------------------------------
# KONFIGURASI KAMERA
# ---------------------------------------------------------
# index 0 = kamera default/bawaan laptop
# index 1, 2, dst = kamera eksternal (USB webcam seperti Jovitek)
CAMERA_INDEX = 1

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Pasangan titik landmark yang digambar sebagai garis (skeleton tangan)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # jempol
    (0, 5), (5, 6), (6, 7), (7, 8),          # telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),     # tengah
    (9, 13), (13, 14), (14, 15), (15, 16),   # manis
    (13, 17), (17, 18), (18, 19), (19, 20),  # kelingking
    (0, 17),
]

TIP_IDS = [4, 8, 12, 16, 20]  # jempol, telunjuk, tengah, manis, kelingking


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print("[INFO] Model belum ada, mendownload hand_landmarker.task ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OK] Model berhasil didownload.")
    except Exception as e:
        print(f"[ERROR] Gagal download model: {e}")
        print("Cek koneksi internet kamu, lalu coba jalankan lagi.")
        raise SystemExit(1)


def create_hand_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def draw_landmarks(frame, hand_landmarks_px):
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, hand_landmarks_px[a], hand_landmarks_px[b], (0, 255, 0), 2)
    for (x, y) in hand_landmarks_px:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def count_fingers(hand_landmarks_px, handedness_label):
    """hand_landmarks_px: list of (x, y) pixel coords, 21 titik.
    handedness_label: 'Left' atau 'Right' (dari sudut pandang kamera, mirror)."""
    fingers = []

    # Jempol: bandingkan posisi x
    if handedness_label == "Right":
        fingers.append(1 if hand_landmarks_px[TIP_IDS[0]][0] > hand_landmarks_px[TIP_IDS[0] - 1][0] else 0)
    else:
        fingers.append(1 if hand_landmarks_px[TIP_IDS[0]][0] < hand_landmarks_px[TIP_IDS[0] - 1][0] else 0)

    # 4 jari lain: bandingkan posisi y (tip lebih tinggi dari pip = terangkat)
    for tip_id in TIP_IDS[1:]:
        fingers.append(1 if hand_landmarks_px[tip_id][1] < hand_landmarks_px[tip_id - 2][1] else 0)

    return fingers.count(1)


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


def main():
    ensure_model_downloaded()
    landmarker = create_hand_landmarker()

    cap, cam_index = open_camera(CAMERA_INDEX)
    if cap is None:
        print("[ERROR] Tidak ada kamera yang terdeteksi. Cek koneksi USB kamera eksternal.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0
    start_time = time.time()

    print("Program berjalan. Tekan 'q' untuk keluar, 'c' untuk ganti kamera.")

    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Gagal membaca frame dari kamera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        total_fingers = 0
        if result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                px_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                draw_landmarks(frame, px_coords)

                handedness_label = "Right"
                if result.handedness and idx < len(result.handedness):
                    handedness_label = result.handedness[idx][0].category_name

                total_fingers += count_fingers(px_coords, handedness_label)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.rectangle(frame, (20, 20), (250, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Jari: {total_fingers}", (35, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Kamera index: {cam_index}", (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Hand Finger Counter", frame)

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

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
