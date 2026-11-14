# -*- coding: utf-8 -*-
"""
ekstrak_video_dataset.py
Skrip otomatis untuk mengubah video rekaman HP menjadi dataset foto koin berkualitas tinggi.
- Mengekstrak 2-3 frame per detik untuk variasi rotasi yang kaya
- Memfilter frame yang buram (motion blur) secara otomatis
- Menyimpan hasil foto ke 'dataset_koin_baru/' siap latih
"""

import os
import sys
import glob
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_RAW_DIR = os.path.join(BASE_DIR, "dataset_video_raw")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "dataset_koin_baru")

VIDEO_EXTENSIONS = ("*.mp4", "*.mov", "*.avi", "*.m4v", "*.3gp", "*.mkv")

# Map folder input ke label target
CLASSES = {
    "100": "100",
    "200": "200",
    "500": "500",
    "1000": "1000",
    "sisi_balik": "sisi_garuda"  # Diselaraskan dengan nama kelas 5-class model
}

def is_frame_sharp(bgr_img, min_lap_var=45.0):
    """Mengecek apakah frame cukup tajam (bebas blur goyangan tangan)."""
    norm = cv2.resize(bgr_img, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return var >= min_lap_var, var

def process_video(video_path, out_folder, target_fps_sample=2.5):
    """Ekstrak frame dari satu video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [GAGAL] Tidak dapat membuka video: {os.path.basename(video_path)}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / target_fps_sample)))

    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    # Bersihkan nama file dari spasi
    safe_name = "".join(c if c.isalnum() else "_" for c in vid_name)

    extracted = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            sharp, lap_var = is_frame_sharp(frame)
            if sharp:
                out_name = f"{safe_name}_f{frame_idx:05d}.png"
                out_path = os.path.join(out_folder, out_name)
                # Tulis file secara aman terhadap path non-ASCII (OneDrive/文档) di Windows
                success, buf = cv2.imencode(".png", frame)
                if success:
                    buf.tofile(out_path)
                    extracted += 1

        frame_idx += 1

    cap.release()
    return extracted

def main():
    print("=" * 72)
    print("      EKSTRAKSI OTOMATIS VIDEO MENJADI DATASET FOTO KOIN RUPIAH")
    print("=" * 72)
    print(f"Folder Sumber: {INPUT_RAW_DIR}")
    print(f"Folder Hasil : {OUTPUT_IMG_DIR}\n")

    total_all_extracted = 0
    summary = {}

    for folder_name, target_label in CLASSES.items():
        src_dir = os.path.join(INPUT_RAW_DIR, folder_name)
        dst_dir = os.path.join(OUTPUT_IMG_DIR, target_label)
        os.makedirs(dst_dir, exist_ok=True)

        # Cari semua video unik di folder ini
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(glob.glob(os.path.join(src_dir, ext)))
        # Hilangkan duplikasi file karena case-insensitive filesystem Windows
        video_files = sorted(list(set(os.path.abspath(f) for f in video_files)))

        print(f"▶ Memproses Kategori '{folder_name}' (Target Label: '{target_label}'):")
        print(f"  Ditemukan {len(video_files)} file video.")

        class_extracted = 0
        for vid in video_files:
            count = process_video(vid, dst_dir)
            print(f"  - {os.path.basename(vid)} -> {count} foto tajam diekstrak.")
            class_extracted += count

        summary[target_label] = class_extracted
        total_all_extracted += class_extracted
        print()

    print("=" * 72)
    print("RINGKASAN HASIL EKSTRAKSI:")
    print("=" * 72)
    for lbl, cnt in summary.items():
        print(f"  Kelas {lbl:<12}: {cnt:>4} foto terkumpul")
    print("-" * 72)
    print(f"  TOTAL FOTO BARU : {total_all_extracted} foto siap latih!")
    print("=" * 72)

    if total_all_extracted == 0:
        print("\n[INFO] Belum ada video yang diproses.")
        print("Silakan masukkan file video (.mp4/.mov) dari HP Anda ke dalam subfolder:")
        print(f"  {INPUT_RAW_DIR}\\100")
        print(f"  {INPUT_RAW_DIR}\\200")
        print(f"  {INPUT_RAW_DIR}\\500")
        print(f"  {INPUT_RAW_DIR}\\1000")
        print(f"  {INPUT_RAW_DIR}\\sisi_balik")
        print("Lalu jalankan skrip ini kembali!")

if __name__ == "__main__":
    main()

