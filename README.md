# Hand Finger Counter (Deteksi Jumlah Jari 1-5)

Program ini mendeteksi tangan lewat kamera (termasuk webcam eksternal seperti
Jovitek di foto kamu) dan menghitung berapa jari yang terangkat, real-time.

Teknologi: **Python + OpenCV + MediaPipe**

---

## 1. Struktur Folder yang Benar

Buat folder project seperti ini (nama folder bebas, contoh `hand-counter`):

```
hand-counter/
├── main.py            <-- kode utama program
├── requirements.txt   <-- daftar library yang dibutuhkan
└── venv/               <-- (akan muncul otomatis setelah langkah 3)
```

Isi persis kedua file (`main.py` dan `requirements.txt`) sudah saya buat,
tinggal download dari chat ini.

---

## 2. Install Python (kalau belum ada)

Cek dulu apakah Python sudah terpasang:

```bash
python --version
```

Kalau belum ada, download dari https://www.python.org/downloads/
Saat instalasi di Windows, **centang "Add Python to PATH"**.

---

## 3. Buat Virtual Environment (opsional tapi disarankan)

Ini supaya library tidak bentrok dengan project lain.

**Windows (PowerShell / CMD):**
```bash
cd hand-counter
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
cd hand-counter
python3 -m venv venv
source venv/bin/activate
```

Kalau berhasil, di depan baris terminal akan muncul tulisan `(venv)`.

---

## 4. Install Library

Masih di dalam folder `hand-counter` (dan venv aktif):

```bash
pip install -r requirements.txt
```

Ini akan menginstall `opencv-python` dan `mediapipe`.

---

## 5. Colokkan Kamera Eksternal (Jovitek)

1. Colokkan kamera USB Jovitek ke laptop.
2. Tutup dulu aplikasi lain yang mungkin pakai kamera (Zoom, Teams, Google Meet, dsb) — kamera hanya bisa dipakai satu program dalam satu waktu.
3. Windows biasanya otomatis install driver-nya. Cek di **Device Manager > Cameras** apakah Jovitek muncul.

**Penting soal index kamera:**
- `0` biasanya = kamera bawaan laptop
- `1` biasanya = kamera eksternal pertama yang dicolok

Di `main.py`, sudah saya set default:
```python
CAMERA_INDEX = 1
```
Kalau nanti kamera eksternal tidak muncul, kamu tidak perlu edit manual —
program otomatis mencoba index 0-4 sampai ketemu. Atau saat program jalan,
tekan tombol **`c`** untuk pindah ke kamera berikutnya.

---

## 6. Jalankan Program

```bash
python main.py
```

Akan muncul jendela kamera. Arahkan tangan ke kamera, jendela akan menampilkan:
- Titik-titik dan garis di tangan (landmark hasil deteksi MediaPipe)
- Angka besar di pojok kiri atas = jumlah jari yang terangkat
- FPS dan index kamera yang sedang dipakai

**Kontrol:**
- Tekan `q` = keluar program
- Tekan `c` = ganti ke kamera lain (kalau kamera eksternal belum kedeteksi)

---

## 7. Cara Kerja Deteksi Jari (ringkas)

MediaPipe mendeteksi 21 titik (landmark) per tangan. Program mengecek:
- **Jempol**: dibandingkan posisi horizontal (x) ujung jempol vs sendi di bawahnya, karena gerak jempol menyamping.
- **4 jari lain** (telunjuk, tengah, manis, kelingking): dianggap "terangkat" kalau posisi ujung jari (tip) secara vertikal (y) lebih tinggi dari sendi tengahnya (pip).

Total jari terangkat dari semua tangan yang terdeteksi (maksimal 2 tangan) dijumlahkan dan ditampilkan di layar.

---

## 8. Troubleshooting

| Masalah | Solusi |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Pastikan `pip install -r requirements.txt` sudah jalan dan venv aktif |
| Jendela kamera hitam/tidak muncul gambar | Tutup aplikasi lain yang pakai kamera, coba tekan `c` untuk ganti index |
| Kamera eksternal tidak terdeteksi sama sekali | Cek kabel USB, coba port USB lain, cek Device Manager |
| Deteksi jari kadang salah hitung | Pastikan pencahayaan cukup terang dan tangan tidak terlalu dekat/jauh dari kamera |
| FPS rendah/lag | Turunkan resolusi di baris `cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)` di `main.py`, misal jadi 640x480 |

---

## 9. Pengembangan Lanjutan (opsional)

Kalau nanti mau dikembangkan lagi, beberapa ide:
- Kirim hasil hitungan jari ke ESP32 lewat MQTT (kamu sudah pernah pakai HiveMQ Cloud di project tangki air, bisa dipakai lagi di sini)
- Tambahkan gesture khusus (misal jari 5 = tutup aplikasi)
- Simpan log jumlah jari ke file CSV untuk analisis

Kalau ada error spesifik pas menjalankan, kirim saja pesan errornya, saya bantu perbaiki.
